"""Canonical identity for rebuilt MJCF; no numeric or geometric fields are relaxed."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


def normalized_xml_sha(path, asset_root, canonical_prefix):
    payload = Path(path).read_text().replace(str(Path(asset_root).resolve()), canonical_prefix)
    root = ET.fromstring(payload)
    for mesh in root.iter('mesh'):
        # MuJoCo may emit this redundant MIME annotation inconsistently after
        # attaching child specs. STL suffix inference selects the same decoder.
        if (mesh.get('content_type') == 'model/stl'
                and Path(mesh.get('file', '')).suffix.lower() == '.stl'):
            del mesh.attrib['content_type']
    canonical = ET.canonicalize(ET.tostring(root, encoding='unicode'), strip_text=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
