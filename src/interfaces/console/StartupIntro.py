"""The startup intro of the TUI (`plans/STARTUP_INTRO_PLAN.md`): once the server's curtains
have closed over the first client's terminal, the Crystal `Scene` plays its flight over the
whole screen — no tab bar, no tab — until the object has settled as the logo. The logo then
holds on the black screen while the title (`Title`) types out to its right with the CRT
cursor, and after a short hold the tabs draw as usual, the Home tab keeping the logo and
the title where they are and the chat starting under the logo, until the transcript fills
and scrolls them up and away. Any key skips to the end of whatever is playing. Later clients of the same process get the tabs straight away."""
from typing import Callable, List, Optional, Tuple

from interfaces.console.crystal.Scene import Scene
from interfaces.console.framework.TUIScreen import TUIScreen, Cell, Color
from interfaces.console.tabs.TabView import CONTENT_TOP
from interfaces.console.widgets.TypewriterText import TypewriterText
from common import Version
from common.logging import info, trace, error

ARMED, FLYING, TITLING, DONE = "armed", "flying", "titling", "done"


class Logo:
    """The settled object as cells: the rectangle of lit cells in the flight's final frame
    rendered on a screen of `size`, kept where that frame put it. Rows above the tab bar's
    spacer are never drawn, so the bar always wins on a terminal too small for both."""

    TOP_ROW = CONTENT_TOP - 1

    def __init__(self, x: int, y: int, cells: List[List[Cell]], size: Tuple[int, int],
                 half_blocks: bool = True):
        self.x = x
        self.y = y
        self.cells = cells
        self.size = size
        self.half_blocks = half_blocks
        self.height = len(cells)
        self.width = len(cells[0]) if cells else 0

    @property
    def bottom(self) -> int:
        return self.y + self.height - 1

    @staticmethod
    def _lit(cell: Cell) -> bool:
        bg = cell.style.bg
        return cell.char != " " or (bg is not None and (bg.r, bg.g, bg.b) != (0, 0, 0))

    @classmethod
    def render(cls, scene: Scene, width: int, height: int, half_blocks: bool = True) -> Optional["Logo"]:
        """The flight's final frame on a `width` × `height` screen, cut down to the lit
        cells; None when nothing is lit (a screen too small to show the object)."""
        scratch = TUIScreen(width, height)
        scratch.half_blocks = half_blocks
        scene.draw_still(scratch, 1, 1, width, height, scene.intro.length())
        lit = [(x, y) for y, row in enumerate(scratch.next_frame) for x, cell in enumerate(row) if cls._lit(cell)]
        if not lit:
            trace(f"StartupIntro: no logo lit on a {width}x{height} screen")
            return None
        x0, x1 = min(x for x, _ in lit), max(x for x, _ in lit)
        y0, y1 = min(y for _, y in lit), max(y for _, y in lit)
        cells = [[Cell(cell.char, cell.style.copy()) for cell in row[x0:x1 + 1]]
                 for row in scratch.next_frame[y0:y1 + 1]]
        logo = cls(x0 + 1, y0 + 1, cells, (width, height), half_blocks)
        info(f"StartupIntro: logo is {logo.width}x{logo.height} cells at ({logo.x}, {logo.y}) on a {width}x{height} screen.")
        return logo

    def blit(self, screen: TUIScreen, lift: int = 0):
        """Draw the logo where it was rendered, `lift` rows higher, clipped to the screen and
        under the tab bar."""
        for row_idx, row in enumerate(self.cells):
            y = self.y + row_idx - lift
            if y < self.TOP_ROW or y > screen.height:
                continue
            target = screen.next_frame[y - 1]
            for col_idx, cell in enumerate(row):
                x = self.x + col_idx
                if 1 <= x <= screen.width:
                    dest = target[x - 1]
                    dest.char = cell.char
                    dest.style.copy_from(cell.style)


class Title:
    """The title typed out to the right of the logo, on its middle row, with the CRT
    cursor sweeping along; once typed the cursor goes and its trail fades. Drawn every
    frame from the landing on, the Home tab included, so the trail fades wherever the
    hand-over happens."""

    TEXT = Version.label()
    CPS = 25          # slow enough to read the typing
    GAP = 3           # cells between the logo's right edge and the text
    LIFT = 0          # rows above the logo's middle row
    COLOR = Color(131, 134, 141)   # the silver piece's grey, bold, so the title reads as one piece with the logo

    def __init__(self, clock: Callable[[], float]):
        self.typewriter = TypewriterText(self.TEXT, cps=self.CPS, clock=clock)

    @property
    def done(self) -> bool:
        return self.typewriter.done

    def skip(self):
        self.typewriter.reveal_all()

    @classmethod
    def position(cls, logo: Logo) -> Tuple[int, int]:
        return logo.x + logo.width + cls.GAP, logo.y + logo.height // 2 - cls.LIFT

    def draw(self, screen: TUIScreen, frame: int, logo: Logo, lift: int = 0):
        x, y = self.position(logo)
        y -= lift
        if y < Logo.TOP_ROW:
            self.typewriter.reset_cursor()   # scrolled under the tab bar: no trail left behind
            return
        visible = self.typewriter.visible_text()
        screen.reset_color()
        screen.set_color(self.COLOR)
        screen.bold()
        screen.write_at(x, y, visible)
        screen.reset_color()
        head = None if self.typewriter.done else (x + len(visible), y)
        self.typewriter.draw_cursor(screen, frame, head, fg=self.COLOR, row_start_x=x,
                                    char_at=lambda cx, cy: screen.row_text(cx, cy, 1) or " ")


class StartupIntro:
    """`draw` is called before the tabs every frame: it draws the flight, then the logo with
    the title typing out, and answers True while the intro owns the screen. `header` draws
    the logo and title for the Home tab afterwards and answers the logo's bottom row."""

    HOLD_SECONDS = 1.0   # the title stays alone on the black screen this long once typed

    def __init__(self, scene: Scene):
        self.scene = scene
        self.state = ARMED
        self.title: Optional[Title] = None
        self._logo: Optional[Logo] = None
        self._typed_at: Optional[float] = None
        self._on_done: List[Callable[[], None]] = []
        self._on_started: List[Callable[[], None]] = []

    def on_started(self, fn: Callable[[], None]):
        """Run `fn` once the intro starts for the first client — right away when it has. An
        opening turn on a local model starts here: its prefill runs behind the flight, and
        the reply lands in the chat pane, which is not drawn until the tabs are."""
        if self.state != ARMED:
            fn()
            return
        self._on_started.append(fn)
        info(f"StartupIntro: {len(self._on_started)} callback(s) waiting for the intro to start.")

    def on_done(self, fn: Callable[[], None]):
        """Run `fn` once the intro is over and the tabs are on screen — right away when they
        already are. The first-run brief waits on this, so the assistant's first reply is
        not sitting there before the chat is shown."""
        if self.state == DONE:
            fn()
            return
        self._on_done.append(fn)
        info(f"StartupIntro: {len(self._on_done)} callback(s) waiting for the intro to end.")

    @property
    def playing(self) -> bool:
        return self.state in (FLYING, TITLING)

    def draw(self, screen: TUIScreen, frame: int) -> bool:
        if self.state == DONE:
            return False
        if self.state == ARMED:
            self.state = FLYING
            self.scene.last_frame_at = None
            self.scene.restart_intro()
            info(f"StartupIntro: the first client is in, flying on a {screen.width}x{screen.height} screen.")
            self._fire(self._on_started, "as the intro started")
            self._on_started = []
        if self.state == FLYING:
            self.scene.draw(screen, 1, 1, screen.width, screen.height)
            if not self.scene.landed:
                return True
            self.state = TITLING
            self.title = Title(self.scene.clock)
            info(f"StartupIntro: landed after {self.scene.flight:.1f}s, typing the title.")
        self.header(screen, frame)
        if self.title.done:
            now = self.scene.clock()
            if self._typed_at is None:
                self._typed_at = now
                info("StartupIntro: title typed, holding.")
            if now - self._typed_at >= self.HOLD_SECONDS:
                self.state = DONE
                info("StartupIntro: over to the tabs.")
                self._fire(self._on_done, "after the intro")
                self._on_done = []
                return False
        return True

    @staticmethod
    def _fire(callbacks: List[Callable[[], None]], when: str):
        for fn in list(callbacks):
            try:
                fn()
            except Exception as e:
                error(f"StartupIntro: a callback failed {when}: {e}")

    def skip(self):
        """A key: the flight jumps to its end, or the title shows whole and the hold ends."""
        if self.state == FLYING:
            self.scene.skip_intro()
        elif self.state == TITLING:
            self.title.skip()
            self._typed_at = self.scene.clock() - self.HOLD_SECONDS
            info("StartupIntro: title skipped.")

    def header(self, screen: TUIScreen, frame: int, lift: int = 0) -> Optional[int]:
        """Draw the logo and the title where the flight left them, `lift` rows higher (the
        Home transcript scrolling them away); nothing once the logo is under the tab bar.
        Answers the logo's bottom row, or None while the flight is still on."""
        logo = self.logo(screen)
        if logo is None:
            return None
        if logo.bottom - lift >= Logo.TOP_ROW:
            logo.blit(screen, lift)
        self.title.draw(screen, frame, logo, lift)
        return logo.bottom

    def logo_bottom(self, screen: TUIScreen) -> Optional[int]:
        """The logo's bottom row on this screen without drawing; None until landed."""
        logo = self.logo(screen)
        return None if logo is None else logo.bottom

    def logo(self, screen: TUIScreen) -> Optional[Logo]:
        """The logo for this screen size; None until the flight has landed."""
        if self.state not in (TITLING, DONE):
            return None
        size = (screen.width, screen.height)
        if self._logo is None or self._logo.size != size or self._logo.half_blocks != screen.half_blocks:
            self._logo = Logo.render(self.scene, *size, half_blocks=screen.half_blocks)
            self.title.typewriter.reset_cursor()
        return self._logo
