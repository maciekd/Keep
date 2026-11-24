import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock

from interfaces.voice.microphones.OutpostServer import OutpostServer


class TestOutpostServer:
        # Should not raise

    def test_broadcast_command_sends_to_all_clients(self):
        server = OutpostServer()

        writer1 = MagicMock()
        writer1.write = Mock()
        writer1.drain = AsyncMock()
        writer1.get_extra_info = Mock(return_value=("127.0.0.1", 1234))

        writer2 = MagicMock()
        writer2.write = Mock()
        writer2.drain = AsyncMock()
        writer2.get_extra_info = Mock(return_value=("127.0.0.1", 5678))

        server.clients = {writer1, writer2}

        asyncio.run(server.broadcast_command("stop"))

        writer1.write.assert_called_once_with(b"<stop>\n")
        writer2.write.assert_called_once_with(b"<stop>\n")

    def test_broadcast_command_removes_disconnected_client(self):
        server = OutpostServer()

        good_writer = MagicMock()
        good_writer.write = Mock()
        good_writer.drain = AsyncMock()
        good_writer.get_extra_info = Mock(return_value=("127.0.0.1", 1111))

        bad_writer = MagicMock()
        bad_writer.write = Mock(side_effect=ConnectionError("disconnected"))
        bad_writer.drain = AsyncMock()
        bad_writer.get_extra_info = Mock(return_value=("127.0.0.1", 2222))
        bad_writer.close = Mock()
        bad_writer.wait_closed = AsyncMock()

        server.clients = {good_writer, bad_writer}

        asyncio.run(server.broadcast_command("stop"))

        assert bad_writer not in server.clients
        assert good_writer in server.clients

    def test_stop_closes_clients_and_server(self):
        server = OutpostServer()
        server.running = True

        writer = MagicMock()
        writer.close = Mock()
        writer.wait_closed = AsyncMock()
        server.clients = {writer}

        mock_server = MagicMock()
        mock_server.close = Mock()
        mock_server.wait_closed = AsyncMock()
        server.server = mock_server

        asyncio.run(server.stop())

        assert server.running is False
        assert len(server.clients) == 0
        writer.close.assert_called_once()
        mock_server.close.assert_called_once()

    def test_handle_client_adds_and_removes_client(self):
        server = OutpostServer()

        reader = MagicMock()
        # Return empty data to trigger disconnect
        reader.read = AsyncMock(return_value=b"")

        writer = MagicMock()
        writer.get_extra_info = Mock(side_effect=lambda key: {
            'peername': ("127.0.0.1", 1234),
            'socket': None
        }.get(key))
        writer.close = Mock()
        writer.wait_closed = AsyncMock()

        asyncio.run(server.handle_client(reader, writer))

        # Client should be removed after disconnect
        assert writer not in server.clients
        writer.close.assert_called()

    def test_broadcast_command_drops_client_on_drain_timeout(self):
        server = OutpostServer()

        good_writer = MagicMock()
        good_writer.write = Mock()
        good_writer.drain = AsyncMock()
        good_writer.get_extra_info = Mock(return_value=("127.0.0.1", 1111))

        slow_writer = MagicMock()
        slow_writer.write = Mock()
        slow_writer.drain = AsyncMock(side_effect=asyncio.TimeoutError())
        slow_writer.get_extra_info = Mock(return_value=("127.0.0.1", 2222))
        slow_writer.close = Mock()
        slow_writer.wait_closed = AsyncMock()

        server.clients = {good_writer, slow_writer}

        asyncio.run(server.broadcast_command("stop"))

        assert slow_writer not in server.clients
        assert good_writer in server.clients
        slow_writer.close.assert_called_once()

    def test_add_command_listener_receives_broadcast_with_no_tcp_clients(self):
        """Local mic outposts subscribe in-process and must receive every
        broadcast even when no TCP outpost is connected. Without this, voice
        commands like 'stop' / 'listen' silently no-op for local mics on a box
        that has no remote outposts attached."""
        server = OutpostServer()
        received = []
        server.add_command_listener(received.append)

        asyncio.run(server.broadcast_command("stop"))

        assert received == ["stop"]

    def test_command_listeners_receive_broadcast_alongside_tcp_clients(self):
        """Listeners aren't a no-TCP-clients fallback; they fire in parallel
        with TCP delivery so a host with both local mics and a remote outpost
        keeps them in sync."""
        server = OutpostServer()
        received = []
        server.add_command_listener(received.append)

        writer = MagicMock()
        writer.write = Mock()
        writer.drain = AsyncMock()
        writer.get_extra_info = Mock(return_value=("127.0.0.1", 1234))
        server.clients = {writer}

        asyncio.run(server.broadcast_command("listen"))

        assert received == ["listen"]
        writer.write.assert_called_once_with(b"<listen>\n")

    def test_resume_reaches_a_tcp_outpost_as_the_wakeword_it_knows(self):
        """`<wakeword>` is a contract with the TCP outposts, which know no `resume`."""
        server = OutpostServer()
        received = []
        server.add_command_listener(received.append)
        writer = MagicMock()
        writer.write = Mock()
        writer.drain = AsyncMock()
        writer.get_extra_info = Mock(return_value=("192.0.2.7", 1234))
        server.clients = {writer}

        asyncio.run(server.broadcast_command("resume"))

        assert received == ["resume"]
        writer.write.assert_called_once_with(b"<wakeword>\n")

    def test_listener_exception_does_not_block_tcp_delivery(self):
        """A buggy listener must not abort the broadcast — TCP outposts still
        need the command. Otherwise a single misbehaving local subscriber
        could silently stop every remote outpost from responding."""
        server = OutpostServer()
        good_listener = Mock()
        bad_listener = Mock(side_effect=RuntimeError("boom"))
        server.add_command_listener(bad_listener)
        server.add_command_listener(good_listener)

        writer = MagicMock()
        writer.write = Mock()
        writer.drain = AsyncMock()
        writer.get_extra_info = Mock(return_value=("127.0.0.1", 1234))
        server.clients = {writer}

        # Must not raise.
        asyncio.run(server.broadcast_command("stop"))

        bad_listener.assert_called_once_with("stop")
        good_listener.assert_called_once_with("stop")
        writer.write.assert_called_once_with(b"<stop>\n")

    def test_handle_client_pong_reports_latency(self):
        from assistant.AssistantEvents import observers
        import time
        server = OutpostServer()

        reader = MagicMock()
        # First read returns pong, second returns empty to disconnect
        reader.read = AsyncMock(side_effect=[b"<pong>", b""])

        writer = MagicMock()
        writer.get_extra_info = Mock(side_effect=lambda key: {
            'peername': ("127.0.0.1", 1234),
            'socket': None
        }.get(key))
        writer.close = Mock()
        writer.wait_closed = AsyncMock()

        # Set a ping timestamp so RTT can be calculated
        server.ping_timestamps[writer] = time.time() - 0.05

        observer = Mock()
        observers().subscribe(observer)

        asyncio.run(server.handle_client(reader, writer))

        observer.on_outpost_latency.assert_called_once()
        args = observer.on_outpost_latency.call_args[0]
        assert args[0] == ("127.0.0.1", 1234)
        assert args[1] > 0  # RTT should be positive
