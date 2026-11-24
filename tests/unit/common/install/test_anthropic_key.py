"""The API key intake of a new install: Anthropic's answers are scripted, the user's typing too."""
from unittest.mock import Mock, patch

import pytest
import requests

from common.install import AnthropicKey as module
from common.install.AnthropicKey import AnthropicKey
from common.install.Terminal import InstallFailed

from .fakes import ScriptedTerminal

WORKSPACE_REQUIRED_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error","message":"anthropic-workspace-id is required '
    'when authenticating with an identity-linked API key; send the id of the workspace this request acts in."},'
    '"request_id":"req_test_0000000000000000"}')
CREDIT_TOO_LOW_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error",'
    '"message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing '
    'to upgrade or purchase credits."}}')
OK = ('200', '')


class Intake(AnthropicKey):
    """An AnthropicKey whose `post` answers from a script (the last answer repeats) and records
    what it was asked about."""

    def __init__(self, answers=None, responses=(OK,), **env):
        super().__init__(ScriptedTerminal(answers), env, '/Users/marcus/Keep')
        self.responses = list(responses)
        self.posted: list[tuple[str, str]] = []

    def post(self, key, workspace):
        self.posted.append((key, workspace))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def screen(intake) -> str:
    return intake.terminal.said


@pytest.mark.unit
class TestObtain:
    def test_environment_key_is_verified_without_a_prompt(self):
        intake = Intake(answers=[], ANTHROPIC_API_KEY='sk-ant-env-47L')
        assert intake.obtain() == ('sk-ant-env-47L', '')
        assert intake.terminal.prompts == []
        assert intake.posted == [('sk-ant-env-47L', '')]
        assert 'verified' in screen(intake)

    def test_rejected_environment_key_stops_the_install(self):
        intake = Intake(responses=[('401', '')], ANTHROPIC_API_KEY='sk-ant-env-47L')
        with pytest.raises(InstallFailed) as failure:
            intake.obtain()
        assert 'ANTHROPIC_API_KEY' in failure.value.reason and 'rejected' in failure.value.reason

    def test_typed_key_is_trimmed_typed_unseen_and_a_malformed_one_never_reaches_anthropic(self):
        intake = Intake(answers=['bogus-key', '  sk-ant-typed-47L  '])
        assert intake.obtain() == ('sk-ant-typed-47L', '')
        assert len(intake.terminal.prompts) == 2 and all(intake.terminal.secret)
        assert intake.posted == [('sk-ant-typed-47L', '')]
        assert 'sk-ant-' in screen(intake), "the slip is not told what a key looks like"

    def test_an_empty_enter_and_a_slip_of_the_paste_use_up_no_attempt(self):
        """Three attempts are three keys Anthropic was asked about."""
        intake = Intake(answers=['', 'sk-ant-one', 'oops', '', 'sk-ant-two', '', 'bogus', 'sk-ant-three'],
                        responses=[('403', ''), ('403', ''), OK])
        assert intake.obtain() == ('sk-ant-three', '')
        assert [key for key, _ in intake.posted] == ['sk-ant-one', 'sk-ant-two', 'sk-ant-three']

    def test_enter_pressed_for_ever_ends_the_install(self):
        intake = Intake(answers=[])
        with pytest.raises(InstallFailed):
            intake.obtain()
        assert intake.posted == []
        assert len(intake.terminal.prompts) == module.SLIPS

    def test_three_rejected_keys_stop_the_install(self):
        intake = Intake(answers=['sk-ant-one', 'sk-ant-two', 'sk-ant-three', 'sk-ant-four'], responses=[('403', '')])
        with pytest.raises(InstallFailed) as failure:
            intake.obtain()
        assert len(intake.terminal.prompts) == 3 and len(intake.posted) == 3
        assert any('ANTHROPIC_API_KEY' in note for note in failure.value.notes), "the way out is not named"

    def test_unverifiable_key_is_kept_with_a_warning(self):
        """Offline or an Anthropic outage must not block the install; the server reports a bad key."""
        intake = Intake(answers=['sk-ant-typed-47L'], responses=[('', '')])
        assert intake.obtain() == ('sk-ant-typed-47L', '')
        assert 'Could not verify' in screen(intake)

    def test_key_without_a_workspace_asks_for_the_id_and_sends_it_from_then_on(self):
        """The 2026-10 beta tester: the Console's Create key dialog left unscoped makes a key
        Anthropic refuses without an anthropic-workspace-id header. The check is a real
        message so this 400 shows at install, not at the first chat."""
        intake = Intake(answers=['sk-ant-typed-47L', 'wrkspc_01TEST'], responses=[('400', WORKSPACE_REQUIRED_BODY), OK])
        assert intake.obtain() == ('sk-ant-typed-47L', 'wrkspc_01TEST')
        said = screen(intake)
        assert 'Scope' in said and 'Organization' in said        # names the dialog's own field
        assert module.KEYS_URL in said and module.WORKSPACES_URL in said
        assert len(intake.terminal.prompts) == 2 and 'Workspace id' in intake.terminal.prompts[1]
        assert intake.terminal.secret == [True, False]
        assert intake.posted == [('sk-ant-typed-47L', ''), ('sk-ant-typed-47L', 'wrkspc_01TEST')]

    def test_enter_instead_of_a_workspace_id_asks_for_a_new_key(self):
        intake = Intake(answers=['sk-ant-one', '', 'sk-ant-two'], responses=[('400', WORKSPACE_REQUIRED_BODY), OK])
        assert intake.obtain() == ('sk-ant-two', '')
        assert len(intake.terminal.prompts) == 3
        assert all(workspace == '' for _, workspace in intake.posted)

    def test_unknown_workspace_id_is_asked_again(self):
        not_found = '{"type":"error","error":{"type":"not_found_error","message":"Workspace `wrkspc_bad` not found."}}'
        intake = Intake(answers=['sk-ant-one', 'wrkspc_bad', 'wrkspc_good'],
                        responses=[('400', WORKSPACE_REQUIRED_BODY), ('404', not_found), OK])
        assert intake.obtain() == ('sk-ant-one', 'wrkspc_good')
        assert 'wrkspc_bad' in screen(intake)
        assert len(intake.terminal.prompts) == 3

    def test_malformed_workspace_id_never_reaches_anthropic(self):
        intake = Intake(answers=['sk-ant-one', 'default', 'wrkspc_01TEST'],
                        responses=[('400', WORKSPACE_REQUIRED_BODY), ('400', WORKSPACE_REQUIRED_BODY), OK])
        assert intake.obtain() == ('sk-ant-one', 'wrkspc_01TEST')
        assert 'default' not in [workspace for _, workspace in intake.posted]
        assert 'wrkspc_' in screen(intake)

    def test_a_workspace_id_never_found_ends_the_install(self):
        intake = Intake(answers=['sk-ant-one'] + ['wrkspc_bad'] * 9, responses=[('400', WORKSPACE_REQUIRED_BODY)])
        with pytest.raises(InstallFailed) as failure:
            intake.obtain()
        assert 'workspace' in failure.value.reason
        assert len(intake.posted) <= 1 + module.WORKSPACE_ATTEMPTS

    def test_environment_workspace_id_is_sent_without_a_prompt(self):
        intake = Intake(answers=[], ANTHROPIC_API_KEY='sk-ant-env-47L', ANTHROPIC_WORKSPACE_ID='wrkspc_01ENV')
        assert intake.obtain() == ('sk-ant-env-47L', 'wrkspc_01ENV')
        assert intake.terminal.prompts == []
        assert intake.posted == [('sk-ant-env-47L', 'wrkspc_01ENV')]
        assert 'wrkspc_01ENV' in screen(intake)

    def test_environment_key_without_a_workspace_names_the_variable_to_set(self):
        intake = Intake(answers=[], responses=[('400', WORKSPACE_REQUIRED_BODY)], ANTHROPIC_API_KEY='sk-ant-env-47L')
        with pytest.raises(InstallFailed) as failure:
            intake.obtain()
        assert any('ANTHROPIC_WORKSPACE_ID' in note for note in failure.value.notes)

    def test_other_400_keeps_the_key_and_repeats_what_anthropic_said(self):
        other = ('{"type":"error","error":{"type":"invalid_request_error",'
                 '"message":"This organization has been disabled."}}')
        intake = Intake(answers=['sk-ant-typed-47L'], responses=[('400', other)])
        assert intake.obtain() == ('sk-ant-typed-47L', '')
        assert 'organization has been disabled' in screen(intake)

    def test_unfunded_organization_waits_for_credit_and_checks_the_same_key_again(self):
        intake = Intake(answers=['sk-ant-typed-47L', ''], responses=[('400', CREDIT_TOO_LOW_BODY), OK])
        assert intake.obtain() == ('sk-ant-typed-47L', '')
        assert module.BILLING_URL in screen(intake)
        assert len(intake.terminal.prompts) == 2 and 'API key' not in intake.terminal.prompts[1]
        assert intake.posted == [('sk-ant-typed-47L', '')] * 2

    def test_skip_keeps_a_key_whose_organization_has_no_credit(self):
        """A new Max subscriber waits 7 days for the plan's credit: the install must still finish."""
        intake = Intake(answers=['sk-ant-typed-47L', ' SKIP '], responses=[('400', CREDIT_TOO_LOW_BODY)])
        assert intake.obtain() == ('sk-ant-typed-47L', '')
        assert len(intake.posted) == 1

    def test_environment_key_without_credit_is_kept_without_a_prompt(self):
        intake = Intake(answers=[], responses=[('400', CREDIT_TOO_LOW_BODY)], ANTHROPIC_API_KEY='sk-ant-env-47L')
        assert intake.obtain() == ('sk-ant-env-47L', '')
        assert intake.terminal.prompts == []

    def test_no_terminal_and_no_environment_key_fails_with_the_workaround(self):
        """curl | bash with no terminal: the install ends saying how to hand the key over."""
        intake = Intake(answers=None)
        with pytest.raises(InstallFailed) as failure:
            intake.obtain()
        assert any('environment' in note for note in failure.value.notes)
        assert intake.posted == []

    def test_the_folder_the_key_is_stored_in_is_named_before_it_is_typed(self):
        intake = Intake(answers=['sk-ant-typed-47L'])
        intake.obtain()
        assert '/Users/marcus/Keep/config.yaml' in screen(intake)


@pytest.mark.unit
class TestTheCheckItself:
    """`post`, the one method that reaches Anthropic — here over a `requests` that answers nothing real."""

    def key(self):
        return AnthropicKey(ScriptedTerminal(), {}, '/Users/marcus/Keep')

    def test_check_is_a_message_so_an_unfunded_organization_shows_at_install(self):
        """A contract with Anthropic's API: the free count_tokens endpoint is not documented to
        check the balance, so a key whose organization has no credit could pass it and leave the
        first chat silent. One token is all that is asked for."""
        with patch.object(requests, 'post', return_value=Mock(status_code=200, text='{}')) as post:
            assert self.key().post('sk-ant-typed-47L', '') == ('200', '{}')
        (url,), sent = post.call_args
        assert url == 'https://api.anthropic.com/v1/messages'
        assert sent['json']['max_tokens'] == 1 and sent['json']['messages']
        assert sent['headers']['x-api-key'] == 'sk-ant-typed-47L'
        assert 'anthropic-workspace-id' not in sent['headers']
        assert sent['timeout'], "a check without a timeout can hold the install for ever"

    def test_the_workspace_id_travels_in_anthropics_header(self):
        with patch.object(requests, 'post', return_value=Mock(status_code=200, text='')) as post:
            self.key().post('sk-ant-typed-47L', 'wrkspc_01TEST')
        assert post.call_args.kwargs['headers']['anthropic-workspace-id'] == 'wrkspc_01TEST'

    def test_no_answer_is_no_status_and_no_exception(self):
        with patch.object(requests, 'post', side_effect=requests.ConnectionError('unreachable')):
            assert self.key().post('sk-ant-typed-47L', '') == ('', '')

    @pytest.mark.parametrize('status, body, workspace, verdict', [
        ('400', WORKSPACE_REQUIRED_BODY, '', module.WORKSPACE),
        ('400', CREDIT_TOO_LOW_BODY, '', module.CREDIT),
        ('400', '{"error":{"message":"The workspace is archived."}}', 'wrkspc_01TEST', module.BAD_WORKSPACE),
        ('400', '{"error":{"message":"The workspace is archived."}}', '', '400'),
        ('404', '{"error":{"message":"Workspace `wrkspc_01TEST` not found."}}', 'wrkspc_01TEST', module.BAD_WORKSPACE),
        ('404', '{"error":{"message":"model not found"}}', 'wrkspc_01TEST', '404'),
        ('529', '<html>overloaded</html>', '', '529'),
    ], ids=['no workspace named', 'no credit', 'a workspace the key cannot use', 'a workspace nobody named',
            'an unknown workspace', 'another 404', 'a body that is no JSON'])
    def test_what_anthropic_answered_becomes_the_verdict(self, status, body, workspace, verdict):
        key = self.key()
        with patch.object(key, 'post', return_value=(status, body)):
            assert key.verify('sk-ant-typed-47L', workspace) == verdict

    def test_the_key_is_not_written_to_the_install_log(self, tmp_path):
        log = tmp_path / 'install.log'
        key = AnthropicKey(ScriptedTerminal(log_path=str(log)), {}, '/Users/marcus/Keep')
        with patch.object(key, 'post', return_value=('401', '')):
            key.verify('sk-ant-typed-47L', '')
        assert 'HTTP 401' in log.read_text() and 'sk-ant-typed-47L' not in log.read_text()
