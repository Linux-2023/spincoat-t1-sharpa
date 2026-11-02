"""One reagent color; target/substrate blues intentionally excluded."""
LIQUID_RGBA = (.83, .25, .09, .78)
LIQUID_OPAQUE_RGBA = (.83, .25, .09, 1.)

def apply_liquid_palette(model):
    """Only named liquid geometries; no materials/lights/robot/target changes."""
    changed = []
    for i in range(model.ngeom):
        name = model.geom(i).name or ''
        if any(token in name for token in ('liquid_visual', 'liquid_proxy', 'tube_bottom_liquid_', 'pour_visual_', 'fluid_', 'film_visual')):
            model.geom_rgba[i, :3] = LIQUID_RGBA[:3]
            changed.append(name)
    return changed
