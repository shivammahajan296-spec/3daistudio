SIMPLE_COSMETIC_JAR_CODE = r'''
import cadquery as cq


def build_model():
    body_radius = 32
    body_height = 52
    wall = 3
    lid_radius = 34
    lid_height = 14

    outer_body = cq.Workplane("XY").circle(body_radius).extrude(body_height)
    inner_cut = (
        cq.Workplane("XY")
        .workplane(offset=4)
        .circle(body_radius - wall)
        .extrude(body_height - 1)
    )
    body = outer_body.cut(inner_cut)

    base_foot = cq.Workplane("XY").circle(body_radius).circle(body_radius - 3).extrude(3)
    neck_ring = (
        cq.Workplane("XY")
        .workplane(offset=body_height - 8)
        .circle(body_radius - 1)
        .circle(body_radius - 5)
        .extrude(5)
    )

    lid = (
        cq.Workplane("XY")
        .workplane(offset=body_height + 2)
        .circle(lid_radius)
        .extrude(lid_height)
    )
    lid_grip = (
        cq.Workplane("XY")
        .workplane(offset=body_height + lid_height + 2)
        .circle(lid_radius - 9)
        .extrude(3)
    )
    label_panel = cq.Workplane("XZ").workplane(offset=-body_radius - 0.4).rect(34, 22).extrude(0.8)

    return body.union(base_foot).union(neck_ring).union(lid).union(lid_grip).union(label_panel)
'''
