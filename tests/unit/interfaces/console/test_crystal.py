import math

import numpy as np
import pytest

from interfaces.console.framework.LinkProbe import LinkStatus
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.TabView import SUB_CONTENT_TOP
from interfaces.console.tabs.CrystalTab import CrystalTab
from interfaces.console.crystal.Scene import Scene
from interfaces.console.crystal.Framebuffer import Framebuffer, UPPER_HALF, terminal_color, focal_length
from interfaces.console.crystal.Intro import Intro, FLY_IN, HOLD, GLIDE, SETTLE, LOGO, approach, ease
from interfaces.console.crystal.Renderer import (Renderer, CUBE, WHITE, AMBIENT, AO_STRENGTH, SKY_STOPS, LIGHTS, Light,
                                                Material, box, solid, environment, normal, dot, rotate, normalized)
from interfaces.console.crystal.Shapes import interlocking_cube, GREEN, SILVER
from interfaces.console.crystal.Tuning import Tuning, PARAMS, LOGO_LOOK

BLACK = (0, 0, 0)


def pixels(fb):
    """Every pixel as an (r, g, b) tuple of 0..255, row by row."""
    return [tuple(px) for px in (fb.color.reshape(-1, 3) * 255 + 0.5).astype(int).tolist()]


def lit_pixels(fb):
    return {(i % fb.width, i // fb.width) for i, v in enumerate(pixels(fb)) if v != BLACK}


def paint(fb, x, y, color):
    fb.color[y, x] = np.asarray(color, np.float32) / 255


def test_fill_polygon_covers_the_inside_and_nothing_else():
    fb = Framebuffer(10, 10)
    fb.fill_polygon([(0, 0), (8, 0), (0, 8)], (200, 100, 50))
    lit = lit_pixels(fb)
    assert (0, 0) in lit and (1, 1) in lit and (7, 0) in lit
    assert (8, 8) not in lit and (9, 0) not in lit
    assert all(x + y <= 7 for x, y in lit)     # pixel centres, not corners, decide
    assert fb.get(1, 1) == (200, 100, 50)
    assert fb.get(9, 9) == BLACK


def test_fill_polygon_takes_quads_wound_either_way():
    for points in ([(2, 2), (6, 2), (6, 5), (2, 5)], [(2, 2), (2, 5), (6, 5), (6, 2)]):
        fb = Framebuffer(8, 8)
        fb.fill_polygon(points, WHITE.color)
        assert lit_pixels(fb) == {(x, y) for x in range(2, 6) for y in range(2, 5)}


def test_fill_polygon_clips_to_the_surface():
    fb = Framebuffer(4, 4)
    fb.fill_polygon([(-10, -10), (20, -10), (-10, 20)], (90, 90, 90))
    assert len(lit_pixels(fb)) == 16


def test_degenerate_polygon_draws_nothing():
    fb = Framebuffer(4, 4)
    fb.fill_polygon([(1, 2), (3, 2), (2, 2)], WHITE.color)
    assert not lit_pixels(fb)
    fb.fill_polygon([(10, 10), (12, 10), (10, 12)], WHITE.color)   # entirely off the surface
    assert not lit_pixels(fb)


def test_rays_pass_through_the_pixel_centres():
    fb = Framebuffer(80, 40)
    focal = focal_length(80, 40)
    assert np.allclose(fb.rays[20, 40], (0.5 / focal, -0.5 / focal, -1))
    assert np.allclose(fb.rays[0, 0], (-39.5 / focal, 19.5 / focal, -1))


def test_rasterize_gives_exact_view_positions_and_keeps_the_nearer_face():
    fb = Framebuffer(40, 40)
    r = Renderer()
    far = [(-1, -1, -6), (1, -1, -6), (1, 1, -6), (-1, 1, -6)]      # a square facing the camera
    near = [(-1, -1, -4), (0, -1, -4), (0, 1, -4), (-1, 1, -4)]     # its left half, nearer
    screen = lambda pts: [r.project(v, 40, 40) for v in pts]
    fragment = fb.rasterize(screen(far), (0, 0, 1), far[0])
    assert np.allclose(fragment.positions[:, 2], -6) and len(fragment.ys) > 100
    assert np.allclose(fb.depth[fragment.ys, fragment.xs], 6)
    fb.paint(fragment, np.tile([1.0, 0.0, 0.0], (len(fragment.ys), 1)))
    fragment = fb.rasterize(screen(near), (0, 0, 1), near[0])
    assert np.allclose(fragment.positions[:, 2], -4)
    assert (fb.rays[fragment.ys, fragment.xs][:, 0] <= 0).all()   # only the left half
    fb.paint(fragment, np.tile([0.0, 0.0, 1.0], (len(fragment.ys), 1)))
    assert fb.rasterize(screen(near), (0, 0, 1), near[0]) is None      # nothing nearer than itself
    assert fb.get(15, 20) == (0, 0, 255) and fb.get(23, 20) == (255, 0, 0)


def test_rasterize_ignores_a_plane_behind_the_camera():
    fb = Framebuffer(20, 20)
    assert fb.rasterize([(0, 0), (20, 0), (20, 20), (0, 20)], (0, 0, 1), (0, 0, 3)) is None


def test_blit_puts_top_pixel_in_foreground_and_bottom_in_background():
    fb = Framebuffer(2, 2)
    paint(fb, 0, 0, (255, 0, 128))   # top-left pixel colored, bottom-left black
    screen = TUIScreen(10, 5)
    fb.blit(screen, 3, 2)
    cell = screen.next_frame[1][2]
    assert cell.char == UPPER_HALF
    assert cell.style.fg is terminal_color(255, 0, 128)
    assert (cell.style.fg.r, cell.style.fg.g, cell.style.fg.b) == (255, 0, 130)   # 64 levels a channel
    assert cell.style.bg is terminal_color(0, 0, 0)
    flat = screen.next_frame[1][3]   # both pixels black: a plain space with a background
    assert flat.char == " " and flat.style.fg is None and flat.style.bg is terminal_color(0, 0, 0)


def test_blit_without_half_blocks_averages_each_pixel_pair_into_a_plain_cell():
    fb = Framebuffer(2, 2)
    paint(fb, 0, 0, (255, 0, 128))   # over black below it
    screen = TUIScreen(10, 5)
    screen.half_blocks = False
    fb.blit(screen, 3, 2)
    cell = screen.next_frame[1][2]
    assert cell.char == " " and cell.style.fg is None
    assert (cell.style.bg.r, cell.style.bg.g, cell.style.bg.b) == (130, 0, 65)


def test_box_winding_points_outward():
    for mesh in (CUBE, box((2, -1, 0.5), (3, 4, 0.7))):
        middle = tuple((lo + hi) / 2 for lo, hi in zip(mesh.lo, mesh.hi))
        for face in mesh.faces:
            assert len(face) == 4
            v0, v1, v2 = (mesh.vertices[i] for i in face[:3])
            centre = tuple(sum(mesh.vertices[i][k] for i in face) / 4 - middle[k] for k in range(3))
            assert dot(normal(v0, v1, v2), centre) > 0


def assert_convex_solid(mesh):
    """Every face planar, convex, wound outward, and every vertex a face uses on or behind its plane.
    A zero-area face — a side squeezed to a ridge by a chamfer at its cap — draws nothing
    and is skipped; returns how many faces were checked."""
    middle = tuple((lo + hi) / 2 for lo, hi in zip(mesh.lo, mesh.hi))
    used = {mesh.vertices[i] for face in mesh.faces for i in face}
    checked = 0
    for face in mesh.faces:
        v = [mesh.vertices[i] for i in face]
        n = normalized(normal(*v[:3]))
        if not any(n):
            continue
        checked += 1
        centre = tuple(sum(p[k] for p in v) / len(v) - middle[k] for k in range(3))
        assert dot(n, centre) > 0, f"{face} wound inward"
        for a, b, c in zip(v, v[1:] + v[:1], v[2:] + v[:2]):
            assert dot(normal(a, b, c), n) > 0, f"{face} concave or misordered"
        for p in used:
            assert dot(n, tuple(p[k] - v[0][k] for k in range(3))) < 1e-9, f"{face} not a supporting plane"
    return checked


def test_chamfered_box_bevels_every_edge_and_caps_every_corner():
    mesh = box((-1, 0, 2), (1, 3, 5), GREEN, chamfer=0.25)
    assert mesh.lo == (-1, 0, 2) and mesh.hi == (1, 3, 5) and mesh.material is GREEN
    assert len(mesh.vertices) == 24 and len(set(mesh.vertices)) == 24
    assert sorted(len(face) for face in mesh.faces) == [3] * 8 + [4] * 18
    assert assert_convex_solid(mesh) == 26
    # Each shrunken face keeps its plane; the bevels sit at 45° to their two faces.
    tilts = sorted(round(max(abs(c) for c in normalized(normal(*(mesh.vertices[i] for i in face[:3])))), 4)
                   for face in mesh.faces)
    assert tilts == [round(1 / math.sqrt(3), 4)] * 8 + [round(1 / math.sqrt(2), 4)] * 12 + [1.0] * 6
    front = [tuple(mesh.vertices[i]) for face in mesh.faces for i in face if len(face) == 4
             and all(mesh.vertices[j][2] == 5 for j in face)]
    assert {(x, y) for x, y, _ in front} == {(-0.75, 0.25), (0.75, 0.25), (0.75, 2.75), (-0.75, 2.75)}


def test_chamfer_is_capped_at_half_the_thinnest_side_and_zero_is_the_plain_box():
    thin = box((0, 0, 0), (4, 4, 1), chamfer=3.0)
    assert thin.lo == (0, 0, 0) and thin.hi == (4, 4, 1)
    assert {v[2] for v in thin.vertices} == {0, 0.5, 1}              # the bevels meet in a ridge halfway up
    assert len(thin.faces) == 26 - 8                # the four sides and their edge bevels squeezed to the ridge
    assert assert_convex_solid(thin) == 18
    plain = box((0, 0, 0), (4, 4, 1), chamfer=0)
    assert plain.vertices == box((0, 0, 0), (4, 4, 1)).vertices and len(plain.faces) == 6


def face_vertices(mesh, axis, value):
    """The vertices of every face of `mesh` lying in the plane `axis` = `value`, rounded."""
    return {tuple(round(x, 6) for x in mesh.vertices[i]) for face in mesh.faces for i in face
            if all(abs(mesh.vertices[j][axis] - value) < 1e-9 for j in face)}


def test_solid_bevels_only_the_convex_edges_of_the_union():
    # Two boxes side by side are one 2 × 1 × 1 slab: their meshes together trace the surface of
    # the single chamfered slab, so the join at x = 1 gets no bevel and every face runs on flat.
    left, right = solid([((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (2, 1, 1))], GREEN, chamfer=0.1)
    assert left.material is right.material is GREEN and left.hi == (1, 1, 1) and right.lo == (1, 0, 0)
    whole = box((0, 0, 0), (2, 1, 1), chamfer=0.1)
    planes = [(normalized(normal(*(whole.vertices[i] for i in face[:3]))), whole.vertices[face[0]])
              for face in whole.faces]
    for mesh in (left, right):
        assert assert_convex_solid(mesh) == len(mesh.faces) == 18      # 6 faces, 8 bevels, 4 caps
        for p in {mesh.vertices[i] for face in mesh.faces for i in face}:
            distances = [dot(n, tuple(p[k] - q[k] for k in range(3))) for n, q in planes]
            assert max(distances) < 1e-9 and min(abs(d) for d in distances) < 1e-9, f"{p} off the slab's surface"
    assert face_vertices(left, 2, 1) == {(0.1, 0.1, 1), (1, 0.1, 1), (1, 0.9, 1), (0.1, 0.9, 1)}
    assert face_vertices(right, 2, 1) == {(1, 0.1, 1), (1.9, 0.1, 1), (1.9, 0.9, 1), (1, 0.9, 1)}


def test_solid_splits_a_box_whose_edge_is_convex_along_only_part_of_its_length():
    # A post standing on a plate: the plate's top edges are real edges only beside the post, so
    # the plate is cut at the post's sides and bevelled outside them, and stays sharp under it.
    # The post's front face runs flat into the plate's, and its vertical edges are bevelled
    # down to the plate's top, where the bevel runs out onto it.
    plate, post = solid([((0, 0, 0), (3, 1, 1)), ((1, 1, 0), (2, 3, 1))], chamfer=0.1)
    assert plate.lo == (0, 0, 0) and plate.hi == (3, 1, 1) and len(plate.vertices) == 3 * 24
    assert len(post.vertices) == 24
    top = face_vertices(plate, 1, 1)
    assert {(1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1)} <= top                          # sharp under the post
    assert {(0.1, 1, 0.1), (1, 1, 0.1), (1, 1, 0.9), (0.1, 1, 0.9)} <= top              # bevelled beside it
    assert {(2, 1, 0.1), (2.9, 1, 0.1), (2.9, 1, 0.9), (2, 1, 0.9)} <= top
    assert not {(0.9, 1, 0.9), (1.1, 1, 0.1), (2.1, 1, 0.1)} & top
    assert face_vertices(post, 2, 1) == {(1.1, 1, 1), (1.9, 1, 1), (1.9, 2.9, 1), (1.1, 2.9, 1)}
    assert {(1, 1, 1), (2, 1, 1), (1, 0.1, 1), (2, 0.1, 1)} <= face_vertices(plate, 2, 1)


def test_camera_is_the_view_offset_with_the_rotation_undone():
    r = Renderer(distance=5.0)
    assert r.camera(0.0, 0.0, 0.0) == (0.0, 0.0, 5.0)
    angles = (0.5, 0.7, 0.3)
    back = rotate(r.camera(*angles), *angles)
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, (0.0, 0.0, 5.0)))


def test_point_on_the_view_axis_projects_to_the_centre():
    r = Renderer()
    assert r.project((0, 0, -5), 80, 40) == (40, 20)


def test_shade_is_ambient_away_from_the_light_and_brightest_facing_it():
    r = Renderer(lights=[Light((0, 0, 0))])             # one white light at the camera
    matte = Material((200, 100, 0))
    towards = r.shade(matte, np.array([[0.0, 0.0, -5.0]]), (0, 0, 1))
    assert np.allclose(towards[0], np.array([200, 100, 0]) / 255 * (1 + AMBIENT))
    away = r.shade(matte, np.array([[0.0, 0.0, -5.0]]), (0, 0, -1))
    assert np.allclose(away[0], np.array([200, 100, 0]) / 255 * AMBIENT)
    sideways = r.shade(matte, np.array([[0.0, 0.0, -5.0]]), (1, 0, 0))
    assert np.allclose(sideways[0], away[0])


def test_shade_adds_each_light_in_its_own_color():
    r = Renderer(lights=[Light((0, 0, 0), (1.0, 0.0, 0.0)), Light((0, 0, 0), (0.0, 0.5, 0.0))])
    rgb = r.shade(Material((255, 255, 255)), np.array([[0.0, 0.0, -5.0]]), (0, 0, 1))
    assert np.allclose(rgb[0], (1 + AMBIENT, 0.5 + AMBIENT, AMBIENT))


def test_shade_varies_across_a_face_and_glossy_faces_get_a_highlight():
    r = Renderer(lights=[Light((-3, 4, -1.5))])
    positions = np.array([[-2.0, 2.0, -5.0], [2.0, -2.0, -5.0]])   # near the light, far from it
    matte = r.shade(Material((255, 255, 255)), positions, (0, 0, 1))
    assert matte[0, 0] > matte[1, 0] + 0.1
    glossy = r.shade(Material((255, 255, 255), gloss=1.0, shininess=10), positions, (0, 0, 1))
    assert (glossy >= matte - 1e-6).all() and glossy[0, 0] > matte[0, 0] + 0.05


def test_environment_blends_from_the_ground_to_the_sky_across_the_horizon():
    ground, zenith = np.array(SKY_STOPS[0][1]), np.array(SKY_STOPS[-1][1])
    assert np.allclose(environment(np.array([-1.0, 1.0])), [ground, zenith])
    assert np.allclose(environment(np.array([0.0]))[0], (ground + zenith) / 2)   # the horizon
    steps = np.diff(environment(np.linspace(-1, 1, 50)), axis=0)
    assert (np.sign(steps) * np.sign(zenith - ground) >= 0).all()   # each channel heads one way


def test_reflective_faces_mirror_the_sky_where_they_tilt_up():
    r = Renderer(lights=[])
    chrome = Material((255, 255, 255), reflectivity=1.0)
    position = np.array([[0.0, 0.0, -5.0]])
    # A face tilted up 45° reflects the camera's ray into the sky; tilted down, into the ground.
    up = r.shade(chrome, position, normalized((0, 1, 1)))
    down = r.shade(chrome, position, normalized((0, -1, 1)))
    assert np.allclose(up[0], SKY_STOPS[-1][1]) and np.allclose(down[0], SKY_STOPS[0][1])
    tinted = r.shade(Material((255, 0, 0), reflectivity=1.0), position, normalized((0, 1, 1)))
    assert np.allclose(tinted[0], (SKY_STOPS[-1][1][0], 0.0, 0.0))
    half = r.shade(Material((255, 255, 255), reflectivity=0.5), position, normalized((0, 1, 1)))
    assert np.allclose(half[0], (np.full(3, AMBIENT) + np.array(SKY_STOPS[-1][1])) / 2)


def test_occlusion_darkens_toward_a_neighbour_that_rises_above_the_face():
    floor = box((-2, -1, -2), (2, 0, 2))
    wall = box((1, 0, -2), (2, 2, 2))            # standing on the floor's right end
    beside = box((-4, -1, -2), (-2, 0, 2))       # against the floor's left end, same height: coplanar top
    r = Renderer([floor, wall, beside], distance=0.0)
    eye = np.eye(3, dtype=np.float32)
    on_floor = np.array([[-1.9, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.0, 0.0, 0.0]])
    light = r.occlusion(floor, on_floor, (0, 1, 0), eye)
    assert light[0] > 0.99                                   # nothing rises above the floor here
    assert (np.diff(light) < 0).all()                        # darker and darker toward the wall
    assert abs(light[-1] - (1 - AO_STRENGTH)) < 1e-6         # fully shadowed against it
    # The wall's inner face is shadowed by the floor at the corner and clear higher up; its top,
    # with nothing rising above it, is untouched.
    inner = r.occlusion(wall, np.array([[1.0, 0.0, 0.0], [1.0, 1.9, 0.0]]), (-1, 0, 0), eye)
    assert abs(inner[0] - (1 - AO_STRENGTH)) < 1e-6 and inner[1] > 0.99
    assert r.occlusion(wall, np.array([[1.5, 2.0, 0.0]]), (0, 1, 0), eye)[0] > 0.99
    assert Renderer([CUBE]).occlusion(CUBE, np.array([[0.0, 1.0, -5.0]]), (0, 1, 0), eye)[0] == 1.0   # alone: no shadow


def test_occlusion_is_the_same_however_the_object_is_turned():
    r = Renderer(interlocking_cube(0.08))
    column = r.meshes[0]
    point = np.array([[0.08, 0.5, 0.5]], np.float32)     # on the column's left face, inside the gap
    still = r.occlusion(column, point - [0, 0, r.distance], (-1, 0, 0), np.eye(3, dtype=np.float32))
    angles = (0.5, 0.7, 0.3)
    axes = r.axes(*angles)
    turned = r.occlusion(column, point @ axes - [0, 0, r.distance], tuple(np.asarray([-1, 0, 0], np.float32) @ axes), axes)
    assert abs(still[0] - turned[0]) < 1e-5
    assert still[0] < 0.7                                 # the block across the gap shadows it


def test_render_culls_half_the_cube_and_lights_the_rest():
    fb = Framebuffer(60, 40)
    drawn = Renderer().render(fb, 0.0, 0.0, 0.0)
    assert drawn == 1          # facing the front square head-on
    assert lit_pixels(fb)
    drawn = Renderer().render(fb, 0.5, 0.7, 0.3)
    assert drawn == 3          # three faces visible from a generic angle


def test_render_paints_a_mesh_in_its_own_color_with_a_gradient():
    fb = Framebuffer(60, 40)
    Renderer([box((-1, -1, -1), (1, 1, 1), Material((0, 200, 0)))]).render(fb, 0.0, 0.0, 0.0)
    shades = {v for v in pixels(fb) if v != BLACK}
    assert len(shades) > 5                       # the point light gives a gradient, not one shade
    assert all(r == b == 0 and 0 < g for r, g, b in shades)     # the rig may push a face past its base


def test_render_never_raises_at_odd_sizes():
    for w, h in ((1, 2), (3, 2), (200, 2), (2, 200)):
        Renderer().render(Framebuffer(w, h), 1.0, 2.0, 3.0)


def test_downsample_averages_each_2x2_block_of_each_channel():
    hi = Framebuffer(4, 2)
    hi.color[0, :, 0] = [1, 1, 1, 0]
    hi.color[1, :, 0] = [1, 1, 0, 0]
    hi.color[0, :, 2] = [0, 0, 0, 1]
    hi.color[1, :, 2] = [0, 0, 1, 1]
    lo = Framebuffer(2, 1)
    hi.downsample_into(lo)
    assert lo.get(0, 0) == (255, 0, 0)
    assert lo.get(1, 0) == (64, 0, 191)       # one red in four, three blue


def test_downsample_4x4_keeps_full_precision():
    hi = Framebuffer(8, 4)
    hi.color[:, :4, 1] = 1.0
    hi.color[:, 4, 1] = 7 / 255
    lo = Framebuffer(2, 1)
    hi.downsample_into(lo)
    assert lo.get(0, 0) == (0, 255, 0) and lo.get(1, 0) == (0, 2, 0)   # 28 / 16 = 1.75


def test_downsample_softens_the_cube_edges():
    hi, lo = Framebuffer(120, 80), Framebuffer(60, 40)
    Renderer().render(hi, 0.5, 0.7, 0.3)
    hi.downsample_into(lo)
    edge = {v[0] for v in pixels(lo) if v != BLACK}
    assert len(edge) > len({v[0] for v in pixels(hi) if v != BLACK})   # blended edge pixels add shades
    assert max(edge) < 256 and min(edge) > 0


def test_blit_sends_neighbouring_cells_of_one_shade_as_a_run_sharing_one_color_object():
    # The screen diff skips a cell when its Color objects are the very same ones as last
    # frame, so colors within one quantization level must map to a single palette entry —
    # otherwise every face is re-sent every frame.
    fb = Framebuffer(4, 2)
    paint(fb, 0, 0, (100, 48, 0))
    paint(fb, 1, 0, (101, 49, 1))
    paint(fb, 2, 0, (100, 48, 200))
    screen = TUIScreen(10, 3)
    fb.blit(screen, 1, 1)
    c0, c1, c2, c3 = (screen.next_frame[0][i] for i in range(4))
    assert c0.style.fg is c1.style.fg
    assert c2.style.fg is not c0.style.fg
    assert c0.char == c1.char == c2.char == UPPER_HALF
    assert c3.char == " " and c3.style.bg is terminal_color(0, 0, 0)
    assert screen.next_frame[0][4].char == " " and screen.next_frame[0][4].style.bg is None


def test_terminal_color_at_a_coarser_depth_snaps_more_colors_onto_one_shade():
    coarse = terminal_color(100, 48, 0, levels=8)
    assert (coarse.r, coarse.g, coarse.b) == (109, 36, 0)          # steps of 255/7
    assert coarse is terminal_color(110, 40, 5, levels=8)          # one step apart at 64, the same at 8
    assert terminal_color(110, 40, 5) is not terminal_color(100, 48, 0)


def test_palette_entries_of_different_depths_never_alias():
    # (3, 1, 0) at 8 levels and (0, 3, 8) at 64 levels pack to the same int; a palette keyed
    # on the int alone answered the second lookup with the first's Color.
    eight = terminal_color(100, 48, 0, levels=8)
    sixty_four = terminal_color(0, 12, 32)
    assert eight is not sixty_four
    assert (sixty_four.r, sixty_four.g, sixty_four.b) == (0, 12, 32)
    assert (eight.r, eight.g, eight.b) == (109, 36, 0)


def test_blit_quantizes_to_the_depth_it_is_given():
    fb = Framebuffer(2, 2)
    paint(fb, 0, 0, (100, 48, 0))
    paint(fb, 1, 0, (110, 40, 5))
    screen = TUIScreen(10, 3)
    fb.blit(screen, 0, 0, levels=8)
    c0, c1 = screen.next_frame[0][0], screen.next_frame[0][1]
    assert c0.style.fg is c1.style.fg is terminal_color(100, 48, 0, levels=8)
    assert c0.style.bg is terminal_color(0, 0, 0, levels=8)


def test_interlocking_cube_pieces_are_disjoint_and_fill_the_cube_when_closed():
    pieces = interlocking_cube(0.0)
    assert sum(math.prod(hi - lo for lo, hi in zip(m.lo, m.hi)) for m in pieces) == 8
    for i, a in enumerate(pieces):
        for b in pieces[i + 1:]:
            assert any(a.hi[k] <= b.lo[k] or b.hi[k] <= a.lo[k] for k in range(3))
    with_gap = interlocking_cube(0.1)
    column, foot, back, block = with_gap
    assert column.material == foot.material == GREEN and back.material == block.material == SILVER
    assert column.lo[0] == column.lo[2] == 0.1 and column.hi == (1, 1, 1)
    assert foot.hi == (0.1, -0.1, 1)                       # touching the column, clear of the block
    assert back.hi[2] == -0.1 and block.lo == (-1, 0.1, -0.1)
    assert all(len(m.faces) > 6 for m in with_gap)                # chamfered by default
    assert all(len(m.faces) == 6 for m in interlocking_cube(0.1, chamfer=0))


def test_interlocking_cube_bevels_each_piece_as_one_solid():
    column, foot, back, block = interlocking_cube(0.1, chamfer=0.05)
    # The L's front face is flat across the column/foot join at x = 0.1, and so is its bottom;
    # its inner vertical edge is bevelled only above the foot.
    front = face_vertices(column, 2, 1) | face_vertices(foot, 2, 1)
    assert {(0.1, -0.95, 1), (0.1, -0.15, 1), (0.1, -0.1, 1), (0.15, -0.1, 1), (0.15, 0.95, 1)} <= front
    assert not {(0.05, -0.95, 1), (0.15, -0.95, 1)} & front
    assert {round(v[0], 6) for v in face_vertices(column, 1, -1) | face_vertices(foot, 1, -1)} == {-0.95, 0.1, 0.95}
    # The silver top is the L outline of back half plus block, inset only along its convex edges.
    top = face_vertices(back, 1, 1) | face_vertices(block, 1, 1)
    assert {(-0.1, 1, -0.1), (-0.1, 1, -0.15), (-0.15, 1, -0.1), (-0.15, 1, 0.95), (0.95, 1, -0.15)} <= top
    assert not {(-0.05, 1, -0.05), (-0.15, 1, -0.15)} & top


def test_chamfered_cube_draws_more_faces_and_keeps_its_silhouette():
    plain, bevelled = Framebuffer(64, 48), Framebuffer(64, 48)
    plain_faces = Renderer(interlocking_cube(0.1, chamfer=0)).render(plain, 0.5, 0.7, 0.1)
    assert Renderer(interlocking_cube(0.1, chamfer=0.05)).render(bevelled, 0.5, 0.7, 0.1) > plain_faces > 0
    inside = lit_pixels(plain) & lit_pixels(bevelled)
    assert len(inside) > 0.9 * len(lit_pixels(plain))              # a small bevel barely moves the outline
    before, after = pixels(plain), pixels(bevelled)
    assert sum(1 for x, y in inside if before[y * 64 + x] != after[y * 64 + x]) > 0.1 * len(inside)


def ray_cast(r, fb, ax, ay, az):
    """The exact picture: per pixel, the nearest box the ray from the camera hits, shaded
    with the renderer's own shader at the hit point."""
    w, h = fb.width, fb.height
    axes = r.axes(ax, ay, az)
    cam = r.camera(ax, ay, az)
    out = [BLACK] * (w * h)
    for py in range(h):
        for px in range(w):
            ray = fb.rays[py, px]
            d = tuple(float(np.dot(a, ray)) for a in axes)
            best = None
            for m in r.meshes:
                tmin, tmax, hit = -1e9, 1e9, None
                for k in range(3):
                    t0, t1 = (m.lo[k] - cam[k]) / d[k], (m.hi[k] - cam[k]) / d[k]
                    n = [0, 0, 0]
                    n[k] = -1 if t0 < t1 else 1
                    t0, t1 = min(t0, t1), max(t0, t1)
                    if t0 > tmin:
                        tmin, hit = t0, tuple(n)
                    tmax = min(tmax, t1)
                if 0 < tmin <= tmax and (best is None or tmin < best[0]):
                    best = (tmin, hit, m)
            if best:
                position = (ray * best[0])[None, :]
                n = rotate(best[1], ax, ay, az)
                rgb = r.shade(best[2].material, position, n) * r.occlusion(best[2], position, n, axes)[:, None]
                out[py * w + px] = tuple(int(c * 255 + 0.5) for c in np.clip(rgb[0], 0, 1))
    return out


def test_interlocking_cube_paints_the_same_picture_as_a_ray_cast():
    # The rasterizer and the ray cast sample the same pixel centres and intersect the same
    # planes, so they agree to rounding everywhere but along edges, where the two decide
    # "inside" with different arithmetic.
    r = Renderer(interlocking_cube(0.08, chamfer=0))
    fb = Framebuffer(36, 24)
    w, h = fb.width, fb.height

    def on_edge(img, x, y):
        around = {img[ny * w + nx] for nx in (x - 1, x, x + 1) for ny in (y - 1, y, y + 1)
                  if 0 <= nx < w and 0 <= ny < h}
        return len(around) > 1

    for angles in ((0.55, -0.7, 0.0), (2.43, 2.64, 1.18), (4.0, 1.0, 5.5), (1.2, 3.9, 2.2), (0.0, 0.0, 0.0)):
        drawn = r.render(fb, *angles)
        exact = ray_cast(r, fb, *angles)
        painted = pixels(fb)
        wrong = [(i % w, i // w) for i, (a, b) in enumerate(zip(painted, exact))
                 if max(abs(p - q) for p, q in zip(a, b)) > 2]
        assert all(on_edge(exact, x, y) for x, y in wrong), f"{angles}: wrong color away from any edge"
        assert len(wrong) < 0.05 * len(exact)
        assert drawn > 2
        assert any(g > r + 40 for r, g, b in painted)                          # green seen
        assert any(max(c) - min(c) < 8 and max(c) > 60 for c in painted)       # silver seen


def tab_rows(screen):
    return ["".join(c.char for c in line).rstrip() for line in screen.next_frame]


def test_crystal_tab_cycles_the_anti_aliasing_factor_on_a():
    tab = CrystalTab()
    seen = [tab.scene.aa]
    for key in "aAaA":
        assert tab.handle_input(key)
        seen.append(tab.scene.aa)
    assert seen == [2, 4, 1, 2, 4]


def test_crystal_tab_cycles_the_color_depth_on_c_and_blits_with_it():
    tab = CrystalTab()
    seen = [tab.scene.color_levels]
    for key in "cCcc":
        assert tab.handle_input(key)
        seen.append(tab.scene.color_levels)
    assert seen == [64, 8, 16, 32, 64]

    def frame_colors(depth):
        while tab.scene.color_levels != depth:
            tab.handle_input('c')
        screen = TUIScreen(120, 20)
        tab.draw(screen, 0)
        cells = [c for row in screen.next_frame[SUB_CONTENT_TOP - 1:-2] for c in row]   # the frame, not the status line
        colors = [cell.style.fg for cell in cells] + [cell.style.bg for cell in cells]
        return screen, [c for c in colors if c is not None]

    eight_shades = {round(k * 255 / 7) for k in range(8)}
    screen, colors = frame_colors(8)
    assert colors, "the cube was drawn"
    assert all({c.r, c.g, c.b} <= eight_shades for c in colors)
    assert "c 8 shades/ch" in tab_rows(screen)[-2]
    _, colors = frame_colors(64)
    assert any(not {c.r, c.g, c.b} <= eight_shades for c in colors), "64 levels use the finer shades"


def test_crystal_tab_leaves_the_navigation_keys_alone_and_space_freezes_the_cube():
    tab = CrystalTab()
    assert tab.handle_input('q') is False and tab.handle_input('shift_up') is False
    tab.draw(TUIScreen(100, 30), 0)
    tab.draw(TUIScreen(100, 30), 1)
    assert tab.scene.tick == 2
    assert tab.handle_input(' ')
    tab.draw(TUIScreen(100, 30), 2)
    assert tab.scene.paused and tab.scene.tick == 2
    screen = TUIScreen(100, 30)
    tab.draw(screen, 3)
    assert "paused" in tab_rows(screen)[-2]


def test_crystal_tab_opens_and_closes_the_gap_on_plus_and_minus():
    tab = CrystalTab()
    assert tab.tuning["gap"] == 0.10
    assert tab.handle_input('+') and tab.tuning["gap"] == 0.12
    assert tab.handle_input('=') and tab.tuning["gap"] == 0.14
    assert tab.scene.renderer.meshes[0].lo[0] == 0.14
    for _ in range(7):
        tab.handle_input('-')
    assert tab.tuning["gap"] == 0.0
    assert tab.scene.renderer.meshes[0].lo[0] == 0.0
    for _ in range(20):
        tab.handle_input('+')                                  # no range: a knob goes wherever it is turned
    assert tab.tuning["gap"] == 0.40
    screen = TUIScreen(120, 30)
    tab.draw(screen, 0)
    assert "gap 0.40" in tab_rows(screen)[-2]


def test_crystal_tab_surfaces_follow_the_terminal_and_the_aa_factor():
    tab = CrystalTab()
    tab.draw(TUIScreen(60, 20), 0)
    rows = 20 - 1 - SUB_CONTENT_TOP
    assert (tab.scene.fb.width, tab.scene.fb.height) == (60, rows * 2)
    assert (tab.scene.hires.width, tab.scene.hires.height) == (120, rows * 4)
    tab.draw(TUIScreen(60, 30), 1)                    # taller only
    rows = 30 - 1 - SUB_CONTENT_TOP
    assert (tab.scene.fb.width, tab.scene.fb.height) == (60, rows * 2)
    assert (tab.scene.hires.width, tab.scene.hires.height) == (120, rows * 4)
    tab.handle_input('a')                             # 2x -> 4x
    tab.draw(TUIScreen(60, 30), 2)
    assert (tab.scene.hires.width, tab.scene.hires.height) == (240, rows * 8)
    tab.handle_input('a')                             # 4x -> off
    tab.draw(TUIScreen(60, 30), 3)
    assert tab.scene.fb.get(0, 0) is not None


def test_crystal_tab_survives_terminals_too_small_for_a_frame():
    tab = CrystalTab()
    for w, h in ((1, 1), (80, SUB_CONTENT_TOP + 1), (2, SUB_CONTENT_TOP + 3), (0, 24)):
        tab.draw(TUIScreen(max(w, 1), max(h, 1)), 0)
    tab.draw(TUIScreen(120, 40), 1)
    assert tab.scene.tick >= 1


def test_tuning_defaults_reproduce_the_module_constants():
    tuning = Tuning()
    assert not any(tuning.changed(p.key) for p in PARAMS)
    assert tuning.lights() == LIGHTS
    assert all(a == b and np.allclose(ca, cb) for (a, ca), (b, cb) in zip(tuning.sky(), SKY_STOPS))
    assert tuning.materials() == (GREEN, SILVER)
    assert len(PARAMS) == len({p.key for p in PARAMS})
    assert all(p.step > 0 for p in PARAMS)


def test_tuning_steps_round_and_reset():
    tuning = Tuning()
    assert tuning.adjust("ambient", 1) == 0.06 and tuning.changed("ambient")
    assert tuning.adjust("ambient", 3) == 0.12                      # no float drift: 0.12, not 0.12000000000000001
    assert tuning.adjust("ambient", -100) == -1.88                  # no range: a knob goes wherever it is turned
    assert tuning.adjust("green_shine", 1) == 45 and tuning.adjust("green_shine", 100) == 545
    assert tuning.reset() == ["Ambient", "Green shininess"]
    assert not tuning.changed("ambient") and tuning.reset() == []


def test_tuning_cursor_wraps_and_nudges_the_selected_param():
    tuning = Tuning()
    assert tuning.current is PARAMS[0]
    assert tuning.select(-1) is PARAMS[-1]
    assert tuning.select(1) is PARAMS[0]
    tuning.select(1)
    param = tuning.nudge(2)
    assert param is PARAMS[1] and math.isclose(tuning[param.key], param.default + 2 * param.step)


def test_tuning_lights_sky_and_materials_follow_the_values():
    tuning = Tuning()
    tuning.set("key_x", 3.0)
    tuning.set("key_power", 2.0)
    tuning.set("fill_power", 0.0)
    key, fill = tuning.lights()
    assert key.position == (3.0, 20.0, 4.5) and key.color == tuple(2 * c for c in LIGHTS[0].color)
    assert fill.color == (0.0, 0.0, 0.0)
    tuning.set("horizon", 0.2)
    tuning.set("ground", 0.2)
    tuning.set("sky", 0.5)
    stops = tuning.sky()
    assert [stop for stop, _ in stops] == [-1.0, -0.2, 0.2, 1.0]
    assert np.allclose(stops[0][1], (0.2, 0.2, 0.24)) and np.allclose(stops[-1][1], (0.5, 0.5, 0.5))
    assert (np.diff(environment(np.linspace(-1, 1, 50), stops), axis=0) >= 0).all()   # horizon stays between them
    tuning.set("ground", 0.0)
    stops = tuning.sky()
    assert np.allclose(stops[0][1], 0)
    assert (0 < np.array(stops[1][1])).all() and (np.array(stops[1][1]) < np.array(stops[2][1])).all()
    assert (np.array(stops[2][1]) < 0.5).all()
    tuning.set("green_mirror", 0.5)
    tuning.set("silver_shine", 10)
    green, silver = tuning.materials()
    assert green.reflectivity == 0.5 and green.color == GREEN.color and silver.shininess == 10


def test_renderer_honours_its_tunable_attributes():
    position = np.array([[0.0, 0.0, -5.0]])
    dark = Renderer(lights=[], ambient=0.5).shade(WHITE, position, (0, 0, -1))
    assert np.allclose(dark[0], 0.5)
    r = Renderer(lights=[])
    r.ambient = 0.25
    assert np.allclose(r.shade(WHITE, position, (0, 0, -1))[0], 0.25)
    r.sky = ((-1.0, (0.3, 0.3, 0.3)), (1.0, (0.3, 0.3, 0.3)))
    assert np.allclose(r.shade(Material((255, 255, 255), reflectivity=1.0), position, normalized((0, 1, 1)))[0], 0.3)
    r.lights = [Light((0, 0, 0), (0.0, 1.0, 0.0))]
    assert np.allclose(r.shade(WHITE, position, (0, 0, 1))[0], (0.25, 1.25, 0.25))
    floor, wall = box((-2, -1, -2), (2, 0, 2)), box((1, 0, -2), (2, 2, 2))
    r = Renderer([floor, wall], distance=0.0, ao_strength=0.2)
    against = np.array([[1.0, 0.0, 0.0]])
    assert abs(r.occlusion(floor, against, (0, 1, 0), np.eye(3, dtype=np.float32))[0] - 0.8) < 1e-6
    r.ao_strength = 0.0
    assert r.occlusion(floor, against, (0, 1, 0), np.eye(3, dtype=np.float32))[0] == 1.0


def test_zoom_scales_the_projection_and_the_rays_together():
    r = Renderer()
    assert abs(r.project((1, 0, -5), 80, 40, 2.0)[0] - 40 - 2 * (r.project((1, 0, -5), 80, 40)[0] - 40)) < 1e-9
    fb = Framebuffer(80, 40, zoom=2.0)
    assert fb.zoom == 2.0 and np.allclose(fb.rays[20, 40], (0.25 / focal_length(80, 40), -0.25 / focal_length(80, 40), -1))
    near, far = Framebuffer(60, 40, zoom=2.0), Framebuffer(60, 40)
    assert len(lit_pixels(near)) == 0
    Renderer().render(near, 0.0, 0.0, 0.0)
    Renderer().render(far, 0.0, 0.0, 0.0)
    assert len(lit_pixels(near)) > 3 * len(lit_pixels(far))


def test_crystal_tab_arrows_pick_and_step_a_param_and_the_renderer_follows():
    tab = CrystalTab()
    assert not tab.panel
    assert tab.handle_input('down') and tab.panel and tab.tuning.current.key == "chamfer"
    assert tab.handle_input('left') and all(len(m.faces) > 6 for m in tab.scene.renderer.meshes)
    assert (1 - 0.015, 1 - 0.015, 1) in tab.scene.renderer.meshes[0].vertices   # inset 0.025 - 0.01
    for _ in range(10):
        tab.handle_input('left')                              # down to 0: the plain boxes
    assert all(len(m.faces) == 6 for m in tab.scene.renderer.meshes)
    assert tab.handle_input('down') and tab.tuning.current.key == "distance"
    assert tab.handle_input('right') and tab.scene.renderer.distance == 5.25
    assert tab.handle_input('left') and tab.handle_input('left') and tab.scene.renderer.distance == 4.75
    tab.handle_input('down')                                  # zoom
    tab.draw(TUIScreen(60, 20), 0)
    assert tab.scene.fb.zoom == 1.0
    tab.handle_input('right')
    tab.draw(TUIScreen(60, 20), 1)
    assert tab.scene.fb.zoom == 1.05 and tab.scene.hires.zoom == 1.05     # the surfaces are rebuilt for the new zoom
    while tab.tuning.current.key != "green_gloss":
        tab.handle_input('down')
    tab.handle_input('right')
    assert tab.scene.renderer.meshes[0].material.gloss == GREEN.gloss + 0.05
    assert tab.scene.renderer.meshes[2].material.gloss == SILVER.gloss
    while tab.tuning.current.key != "key_y":
        tab.handle_input('up')
    tab.handle_input('left')
    assert tab.scene.renderer.lights[0][0][1] == LIGHTS[0].position[1] - 0.5
    assert tab.handle_input('r')
    assert tab.scene.renderer.distance == 5.0 and tab.scene.renderer.meshes[0].material == GREEN
    assert np.allclose(tab.scene.renderer.lights[0][0], LIGHTS[0].position)
    tab.draw(TUIScreen(60, 20), 2)
    assert tab.scene.fb.zoom == 1.0


def test_crystal_tab_spin_speed_scales_the_rotation_and_zero_freezes_it():
    tab = CrystalTab()
    while tab.tuning.current.key != "spin":
        tab.handle_input('down')
    for _ in range(10):
        tab.handle_input('left')
    assert tab.tuning["spin"] == 0.0
    screen = TUIScreen(60, 20)
    tab.draw(screen, 0)
    first = pixels(tab.scene.fb)
    tab.draw(screen, 1)
    assert pixels(tab.scene.fb) == first                            # not turning, though the tick advances
    assert tab.scene.tick == 2


def test_crystal_tab_panel_lists_the_params_and_marks_the_selection_and_changes():
    tab = CrystalTab()
    screen = TUIScreen(80, 60)                                # tall enough for the whole list
    tab.draw(screen, 0)
    assert not any("Tuning" in row for row in tab_rows(screen))
    tab.handle_input('down')
    tab.handle_input('right')
    screen = TUIScreen(80, 60)
    tab.draw(screen, 1)
    rows = tab_rows(screen)
    assert " Tuning" in rows[CrystalTab.PANEL_Y - 1]
    assert "  Gap" in rows[CrystalTab.PANEL_Y] and "0.10" in rows[CrystalTab.PANEL_Y]
    assert " >Chamfer" in rows[CrystalTab.PANEL_Y + 1] and "0.035*" in rows[CrystalTab.PANEL_Y + 1]
    assert "  Silver mirror" in rows[CrystalTab.PANEL_Y + len(PARAMS) - 1]
    assert tab.handle_input('t') and not tab.panel
    screen = TUIScreen(80, 60)
    tab.draw(screen, 2)
    assert not any("Tuning" in row for row in tab_rows(screen))


def test_crystal_tab_panel_scrolls_to_keep_the_selection_in_a_short_terminal():
    tab = CrystalTab()
    for _ in range(len(PARAMS) - 1):
        tab.handle_input('down')                              # the last param
    screen = TUIScreen(80, 12)
    tab.draw(screen, 0)
    rows = tab_rows(screen)
    assert any(" >Silver mirror" in row for row in rows)
    assert not any("Gap" in row for row in rows)
    tab.draw(TUIScreen(80, SUB_CONTENT_TOP + 3), 1)              # barely a row for the panel: no crash


def test_look_at_pans_the_rays_and_the_projection_together():
    fb = Framebuffer(120, 60, zoom=1.5)
    assert fb.centre == (60.0, 30.0) and fb.project((0, 0, -5)) == (60.0, 30.0)
    rays = fb.rays.copy()
    fb.look_at(0.12, 0.2)
    assert np.allclose(fb.centre, (14.4, 12.0)) and np.allclose(fb.project((0, 0, -7)), (14.4, 12.0))
    assert np.allclose(fb.rays[12, 14], (0.1 / fb.focal, -0.5 / fb.focal, -1)) and not np.allclose(fb.rays, rays)
    sx, sy = fb.project((1.0, -0.5, -4.0))
    assert np.allclose(fb.rays[int(sy), int(sx)][:2] * 4.0, (1.0, -0.5), atol=0.5 * 4.0 / fb.focal)
    rays = fb.rays
    fb.look_at(0.12, 0.2)                                    # unchanged: nothing rebuilt
    assert fb.rays is rays


def test_panning_the_principal_point_moves_the_picture_without_reshaping_it():
    centred, panned = Framebuffer(80, 40), Framebuffer(80, 40)
    panned.look_at(0.25, 0.5)
    r = Renderer()
    r.render(centred, 0.3, 0.5, 0.1)
    r.render(panned, 0.3, 0.5, 0.1)
    shifted = {(x + 20, y) for x, y in lit_pixels(panned)}
    assert shifted == lit_pixels(centred)                     # the same picture, 20 px to the left
    assert np.allclose(panned.color[:, :60], centred.color[:, 20:])


def test_ease_and_approach_shapes():
    assert ease(-1) == 0 and ease(0) == 0 and ease(0.5) == 0.5 and ease(1) == 1 and ease(2) == 1
    assert approach(300, 2.5, 0, 1.0) == 300 and math.isclose(approach(300, 2.5, 1, 1.0), 2.5)
    assert math.isclose(approach(300, 2.5, 0.5, -1.0), 151.25)                   # halfway in distance
    assert math.isclose(approach(300, 2.5, 0.5, 0.0), math.sqrt(300 * 2.5))         # halfway in the size's logarithm
    assert math.isclose(1 / approach(300, 2.5, 0.5, 1.0), (1 / 300 + 1 / 2.5) / 2)   # halfway in size
    assert (approach(300, 2.5, 0.5, 1.0) < approach(300, 2.5, 0.5, 0.5) < approach(300, 2.5, 0.5, 0.0)
            < approach(300, 2.5, 0.5, -1.0))


def test_intro_flies_in_holds_glides_to_the_corner_and_stays_as_the_logo():
    tuning = Tuning()
    intro = Intro(tuning.__getitem__)
    assert intro.length() == 3.5 + 3.0 + 3.0 + 2.0                      # fly-in, hold, glide, settle
    start = intro.shot(0.0)
    assert start.stage == FLY_IN and start.distance == 500 and (start.x, start.y, start.spin) == (0.5, 0.5, 1.0)
    distances = [intro.shot(t).distance for t in np.linspace(0, 3.5, 36)]
    assert all(a > b for a, b in zip(distances, distances[1:]))                 # ever nearer
    assert math.isclose(intro.shot(3.5).distance, 3.7) and intro.shot(3.5).stage == HOLD
    assert intro.shot(6.4).stage == HOLD and math.isclose(intro.shot(6.4).distance, 3.7)
    mid = intro.shot(6.5 + 1.5)
    assert mid.stage == GLIDE and 3.7 < mid.distance < 10.5
    assert math.isclose(mid.x, (0.5 + 0.10) / 2) and math.isclose(mid.y, (0.5 + 0.17) / 2)
    assert math.isclose(mid.spin, (1.0 + 0.3) / 2)
    settling = intro.shot(10.5)
    assert settling.stage == SETTLE and 0 < settling.spin < 0.3 and math.isclose(settling.distance, 10.5)
    logo = intro.shot(11.5)
    assert logo.stage == LOGO and math.isclose(logo.distance, 10.5) and logo.spin == 0.0
    assert math.isclose(logo.x, 0.10) and math.isclose(logo.y, 0.17)
    assert intro.shot(60.0) == logo
    tuning.set("logo_x", 0.9)
    tuning.set("fly_seconds", 1.0)
    assert intro.shot(60.0).x == 0.9 and intro.shot(1.0).stage == HOLD        # the knobs act at once


def test_crystal_tab_plays_the_intro_on_real_time_and_i_and_o_restart_and_stop_it():
    now = [100.0]
    tab = CrystalTab(Scene(clock=lambda: now[0]))
    scene = tab.scene
    screen = TUIScreen(120, 40)
    tab.draw(screen, 0)
    assert scene.shot is None and "orbit" in tab_rows(screen)[-2] and scene.renderer.distance == 5.0
    tab.on_activate()
    tab.draw(TUIScreen(120, 40), 1)
    first = Scene.FRAME_SECONDS                                # the first frame after a restart counts as one frame
    assert scene.flight == pytest.approx(first) and scene.shot.stage == FLY_IN
    now[0] += 2.0                                              # a late frame counts as a quarter second
    tab.draw(TUIScreen(120, 40), 2)
    assert scene.flight == pytest.approx(first + 0.25)
    for _ in range(12):                                        # quarter-second frames on one row of picture: only the clock matters
        now[0] += 0.25
        tab.draw(TUIScreen(8, SUB_CONTENT_TOP + 2), 3)
    assert scene.flight == pytest.approx(first + 3.25) and scene.shot.stage == FLY_IN
    tab.handle_input(' ')                                      # paused: the flight stands still
    now[0] += 1.0
    tab.draw(TUIScreen(120, 40), 4)
    assert scene.flight == pytest.approx(first + 3.25)
    tab.handle_input(' ')
    scene.skip_intro()                                         # the rest of the flight is the Intro's own test
    screen = TUIScreen(120, 40)
    tab.draw(screen, 6)
    assert scene.shot.stage == LOGO and scene.landed and math.isclose(scene.renderer.distance, 10.5)
    lx, ly = scene.logo_at()   # the tuning's (0.10, 0.17), pushed in where the surface is too small for the logo there
    assert lx >= 0.10 and ly >= 0.17
    assert np.allclose(scene.fb.centre, (lx * scene.fb.width, ly * scene.fb.height))
    assert np.allclose(scene.hires.centre, (lx * scene.hires.width, ly * scene.hires.height))
    assert "intro logo" in tab_rows(screen)[-2]
    lit = lit_pixels(scene.fb)
    assert max(x for x, _ in lit) < scene.fb.width / 3 and max(y for _, y in lit) < scene.fb.height / 2
    assert tab.handle_input('i') and scene.flight == 0.0
    assert tab.handle_input('o') and scene.flight is None
    tab.draw(TUIScreen(120, 40), 7)
    assert scene.shot is None and tab.scene.renderer.distance == 5.0 and tab.scene.fb.centre == (tab.scene.fb.width / 2, tab.scene.fb.height / 2)


def test_crystal_tab_tumble_never_jumps_when_the_spin_changes():
    now = [100.0]
    tab = CrystalTab(Scene(clock=lambda: now[0]))
    tab.draw(TUIScreen(60, 20), 0)                            # the first frame counts as one frame of tumble
    now[0] += Scene.FRAME_SECONDS
    tab.draw(TUIScreen(60, 20), 1)
    assert tab.scene.turn == pytest.approx(2.0)
    tab.tuning.set("spin", 0.5)
    now[0] += Scene.FRAME_SECONDS
    tab.draw(TUIScreen(60, 20), 2)
    assert tab.scene.turn == pytest.approx(2.5)


def test_intro_fades_the_look_and_the_pose_to_the_logo_over_the_last_seconds():
    tuning = Tuning()
    intro = Intro(tuning.__getitem__, lambda key: tuning.value(key, 1.0))
    end, fade = intro.length(), tuning["fade_seconds"]
    assert intro.fade(end - fade) == 0.0 and intro.fade(end - fade / 2) == 0.5 and intro.fade(end) == 1.0 == intro.fade(60.0)
    assert 0.0 < intro.fade(end - fade * 0.9) < 0.1 / 2                                # eased in, not a straight line
    assert np.allclose(np.degrees(intro.angles(end)), (LOGO_LOOK["final_x"], LOGO_LOOK["final_y"], LOGO_LOOK["final_z"]))
    assert not np.allclose(np.degrees(intro.angles(end - fade)), (-46, 80, 180))      # the tumble is still winding down
    assert np.allclose(np.degrees(Intro(tuning.__getitem__).angles(end)), (-46, 80, 180))   # without a logo look the pose stays
    assert tuning.faded(0.0) == tuning.values
    half = tuning.faded(0.5)
    assert half["ambient"] == pytest.approx((0.04 + 0.76) / 2) and half["gap"] == tuning["gap"]
    assert tuning.materials(1.0)[0].shininess == 90 and tuning.lights(1.0)[0].position[0] == 61.5
    tuning.set("fade_seconds", 2.0)
    assert intro.fade(end - 1.0) == 0.5
    tuning.set("ease", 0.0)
    assert intro.fade(end - 1.8) == pytest.approx(0.1)                                 # ease 0: linear


def test_scene_pushes_the_faded_look_into_the_renderer_and_the_orbit_takes_it_back():
    now = [100.0]
    scene = Scene(clock=lambda: now[0])
    screen = TUIScreen(80, 30)
    scene.restart_intro()
    end, fade = scene.intro.length(), scene.tuning["fade_seconds"]

    def frame_at(flight):
        """One frame with the flight's clock put at `flight`; the frames in between would only move it there."""
        scene.flight = flight
        scene.draw(screen, 1, 1, 80, 29)

    frame_at(end - fade - 0.5)
    assert scene.look_fade == 0.0 and scene.renderer.ambient == 0.04
    frame_at(end - fade * 0.4)
    assert 0.0 < scene.look_fade < 1.0 and 0.04 < scene.renderer.ambient < 0.76
    assert 1.5 > scene.renderer.meshes[0].material.gloss > 0.4
    frame_at(end)
    assert scene.landed
    assert scene.look_fade == 1.0 and scene.renderer.ambient == 0.76 and scene.renderer.ao_radius == 2.2
    assert scene.renderer.meshes[0].material.shininess == 90 and scene.renderer.meshes[-1].material.reflectivity == 0.12
    assert np.allclose(scene.renderer.lights[0][0], (61.5, 67.5, -4.5))
    scene.tuning.set("gap", 0.2)
    scene.apply()                                                       # a knob keeps the look where the flight is
    assert scene.look_fade == 1.0 and scene.renderer.ambient == 0.76
    scene.stop_intro()
    now[0] += 0.1
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.look_fade == 0.0 and scene.renderer.ambient == 0.04 and scene.renderer.meshes[0].material == GREEN
    scene.draw_still(screen, 1, 1, 80, 29, scene.intro.length())      # the logo is drawn with the logo's look
    assert scene.renderer.ambient == 0.76
    now[0] += 0.1
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.renderer.ambient == 0.04


def test_scene_drops_the_shades_the_frame_the_link_falls_behind_and_asks_for_probes():
    scene = Scene(clock=lambda: 100.0)
    screen = TUIScreen(80, 30)
    scene.draw(screen, 1, 1, 80, 29)
    assert screen.link_probe_wanted, "the live picture wants the link measured wherever it is drawn"
    assert scene.color_levels == 64
    screen.link_status = LinkStatus(lag=Scene.DROP_LAG / 2, saturated=True)      # a red status line, not yet a stutter
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.color_levels == 64
    screen.link_status = LinkStatus(lag=Scene.DROP_LAG, saturated=True)
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.color_levels == 8 and scene.lagging
    scene.color_levels = 64                                          # cycled back by hand while it still lags
    screen.link_status = LinkStatus(lag=3.0, saturated=True)
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.color_levels == 64, "one drop per episode"
    screen.link_status = LinkStatus(lag=0.05)
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.color_levels == 64 and not scene.lagging, "recovery leaves the depth alone"
    screen.link_status = LinkStatus(lag=1.2, saturated=True)
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.color_levels == 8, "a new episode drops it again"
    screen.link_status = LinkStatus(unsupported=True)
    scene.draw(screen, 1, 1, 80, 29)
    assert scene.color_levels == 8 and not scene.lagging


def test_crystal_tab_arrows_tune_the_logo_look_once_the_intro_has_landed():
    now = [100.0]
    tab = CrystalTab(Scene(clock=lambda: now[0]))
    scene, tuning, screen = tab.scene, tab.tuning, TUIScreen(80, 30)
    scene.restart_intro()
    tab.draw(screen, 0)
    assert not tab.editing_logo
    scene.skip_intro()
    tab.draw(screen, 1)
    assert tab.editing_logo and scene.renderer.ambient == LOGO_LOOK["ambient"]
    tuning.selected = [p.key for p in PARAMS].index("ambient")
    tab.handle_input('right')
    assert scene.renderer.ambient == pytest.approx(LOGO_LOOK["ambient"] + 0.02)      # the picture follows the knob
    assert tuning["ambient"] == 0.04 and tuning.changed("ambient", logo=True) and not tuning.changed("ambient")
    assert tuning.shown("ambient", logo=True) == scene.renderer.ambient
    tuning.selected = 0                                                               # gap has no logo value
    tab.handle_input('right')
    assert tuning["gap"] == pytest.approx(0.12) and tuning.shown("gap", logo=True) == tuning["gap"]
    tab.handle_input('o')
    assert not tab.editing_logo
    assert tuning.reset() == ["Gap", "Ambient"] and tuning.logo == LOGO_LOOK
