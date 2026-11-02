"""Query-only MuJoCo 3.6.0 CCD allocator repair for isolated sweep planning.

Use ``with prepare_sweep_distance(m, work, scratch_parent=owned_run_directory)
 as distance:`` once per sweep; call ``distance(g1, g2, distmax, fromto=None)``.
``work`` must be planning-only MjData, never live simulation data. The bound
model/data must not be used concurrently or mutated by another thread.

Only the missing mjCCDConfig allocator fields are repaired. Native dispatch,
GJK/EPA, iteration count, tolerance, witness order, flags and distance clipping
are preserved. Unaffected dispatch calls the original mj_geomDistance; that is
not an error fallback. Importing this module does not compile or patch anything.

The private ABI is restricted to the observed Linux x86_64 double-precision
3.6.0 library and installed header hashes. Other builds reject. A private 0700
TemporaryDirectory below the caller's existing sandbox/output directory owns
only a verified copy of the prebuilt query library; context exit removes it.
No runtime compiler, subprocess exception, shared cache or library replacement.
The runtime metadata describes the implementation, not a validation result.
"""
import base64
import ctypes
import hashlib
import json
import operator
import os
from pathlib import Path
import platform
import sys
import tempfile

import mujoco
import numpy as np


_REPAIR_ID = 'mujoco_3.6.0_mj_geomDistanceCCD_allocator_init'
_LIBRARY_SHA256 = '385ae719dde65acffdb0a66605e43fa0c680cb8b331fc4dfe26b39ab3f5630ff'
_HEADERS_SHA256 = 'b08ecfd7ee4f1a1d832e0992e635ae6caa438ad334799e1f6adf8a40b730d9fc'
_SOURCE_ROOT = 'https://raw.githubusercontent.com/google-deepmind/mujoco/3.6.0/src/engine/'
_SOURCE_SHA256 = {
    'engine_support.c': '8bf7befb3241a3aad2f3318825ee89c42fe443c52c246cf0e77c6aaf45af7e9b',
    'engine_collision_gjk.h': 'eacd6022490af5d41d167ce899061f8de11cd300e26d0ce2568f9fcc29382055',
    'engine_collision_convex.h': '92b8300dda6bbb904339ecc92a4c3c079d753a7dc7ce9afb25a31a5e5e879f6e',
    'engine_collision_convex.c': 'd9be203c18f56e9af3db080fecdc5348bcc90a9bce80561dd62549f50e26b18c',
    'engine_collision_primitive.h': 'a6c2b9d8c5bf0d612f16a15863c65f23eae244ae4606489fc00eead00847d54b',
}

_C_SOURCE = r'''
// Copyright 2021, 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Derived from the official MuJoCo 3.6.0 sources identified in _SOURCE_SHA256.
// Modifications: embed the exact needed private ABI declarations, add ABI and
// loader-identity guards, dispatch unaffected queries to the original public
// function, and initialize context/alloc/free using the native collision
// allocator. No GJK/EPA implementation or physics code is replaced.
#include <stddef.h>
#include <mujoco/mujoco.h>

#if mjVERSION_HEADER != 3006000 || defined(mjUSESINGLE)
#error "This query repair requires MuJoCo 3.6.0 double-precision headers"
#endif

// engine_collision_convex.h: private object layout, unchanged.
struct _mjCCDObj {
  const mjModel* model;
  const mjData* data;
  int geom;
  int geom_type;
  int vertindex;
  int meshindex;
  int flex;
  int elem;
  int vert;
  mjtNum margin;
  mjtNum rotate[4];
  void (*center)(mjtNum res[3], const struct _mjCCDObj* obj);
  void (*support)(mjtNum res[3], struct _mjCCDObj* obj, const mjtNum dir[3]);
  mjtNum prism[6][3];
  const mjtNum* size;
  const float* hfield_data;
  int hfield_nrow;
  int hfield_ncol;
};
typedef struct _mjCCDObj mjCCDObj;

// engine_collision_gjk.h: private declarations, unchanged.
typedef enum {
  mjEPA_NOCONTACT = -1,
  mjEPA_SUCCESS = 0,
  mjEPA_P2_INVALID_FACES,
  mjEPA_P2_NONCONVEX,
  mjEPA_P2_ORIGIN_ON_FACE,
  mjEPA_P3_BAD_NORMAL,
  mjEPA_P3_INVALID_V4,
  mjEPA_P3_INVALID_V5,
  mjEPA_P3_MISSING_ORIGIN,
  mjEPA_P3_ORIGIN_ON_FACE,
  mjEPA_P4_MISSING_ORIGIN,
} mjEPAStatus;

typedef struct {
  mjtNum vert[3];
  mjtNum vert1[3];
  mjtNum vert2[3];
  int index1;
  int index2;
} Vertex;

typedef struct {
  int max_iterations;
  mjtNum tolerance;
  int max_contacts;
  mjtNum dist_cutoff;
  void* context;
  void*(*alloc)(void* context, size_t nbytes);
  void(*free)(void* context, void* buffer);
} mjCCDConfig;

typedef struct {
  mjtNum dist;
  mjtNum x1[3 * mjMAXCONPAIR];
  mjtNum x2[3 * mjMAXCONPAIR];
  int nx;
  int max_iterations;
  mjtNum tolerance;
  int max_contacts;
  mjtNum dist_cutoff;
  int gjk_iterations;
  int epa_iterations;
  mjEPAStatus epa_status;
  Vertex simplex[4];
  int nsimplex;
} mjCCDStatus;

MJAPI void mjc_initCCDObj(mjCCDObj* obj, const mjModel* m, const mjData* d,
                         int g, mjtNum margin);
MJAPI mjtNum mjc_ccd(const mjCCDConfig* config, mjCCDStatus* status,
                    mjCCDObj* obj1, mjCCDObj* obj2);
MJAPI int mjc_Convex(const mjModel* m, const mjData* d,
                    mjContact* con, int g1, int g2, mjtNum margin);
// engine_collision_primitive.h; same collision callback signature.
MJAPI int mjc_BoxBox(const mjModel* m, const mjData* d,
                    mjContact* con, int g1, int g2, mjtNum margin);

_Static_assert(sizeof(void*) == 8 && sizeof(int) == 4 && sizeof(mjtNum) == 8,
               "Unexpected public scalar ABI");
_Static_assert(mjMAXCONPAIR == 50 && sizeof(mjEPAStatus) == 4 && sizeof(Vertex) == 80,
               "Unexpected private scalar ABI");
_Static_assert(sizeof(mjCCDObj) == 272 && offsetof(mjCCDObj, margin) == 48 &&
               offsetof(mjCCDObj, center) == 88 && offsetof(mjCCDObj, prism) == 104 &&
               offsetof(mjCCDObj, hfield_ncol) == 268, "Unexpected mjCCDObj ABI");
_Static_assert(sizeof(mjCCDConfig) == 56 && offsetof(mjCCDConfig, context) == 32 &&
               offsetof(mjCCDConfig, alloc) == 40 && offsetof(mjCCDConfig, free) == 48,
               "Unexpected mjCCDConfig ABI");
_Static_assert(sizeof(mjCCDStatus) == 2784 && offsetof(mjCCDStatus, nx) == 2408 &&
               offsetof(mjCCDStatus, dist_cutoff) == 2432 &&
               offsetof(mjCCDStatus, simplex) == 2456 &&
               offsetof(mjCCDStatus, nsimplex) == 2776, "Unexpected mjCCDStatus ABI");

// engine_collision_convex.c: identical stack allocation and alignment.
static void* ccd_allocate(void* data, size_t nbytes) {
  mj_markStack((mjData*)data);
  return mj_stackAllocByte((mjData*)data, nbytes, sizeof(mjtNum));
}

static void ccd_free(void* data, void* buffer) {
  (void)buffer;
  mj_freeStack((mjData*)data);
}

// engine_support.c: mj_geomDistanceCCD, with only its missing fields added.
static mjtNum repaired_distance_ccd(const mjModel* m, const mjData* d,
                                    int g1, int g2, mjtNum distmax, mjtNum fromto[6]) {
  mjCCDConfig config;
  mjCCDStatus status;
  config.max_iterations = m->opt.ccd_iterations;
  config.tolerance = m->opt.ccd_tolerance;
  config.max_contacts = 1;
  config.dist_cutoff = distmax;
  config.context = (void*)d;
  config.alloc = ccd_allocate;
  config.free = ccd_free;

  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m, d, g1, 0);
  mjc_initCCDObj(&obj2, m, d, g2, 0);
  mjtNum dist = mjc_ccd(&config, &status, &obj1, &obj2);
  if (fromto && status.nx > 0) {
    mju_copy3(fromto, status.x1);
    mju_copy3(fromto+3, status.x2);
  }
  return dist < distmax ? dist : distmax;
}

// Match the original dispatch predicate. All other paths use the exact native
// primitive/libccd implementation, not a substitute or an error fallback.
mjtNum v34_sweep_geom_distance(const mjModel* m, const mjData* d,
                               int geom1, int geom2, mjtNum distmax, mjtNum fromto[6]) {
  int flip = m->geom_type[geom1] > m->geom_type[geom2];
  int g1 = flip ? geom2 : geom1;
  int g2 = flip ? geom1 : geom2;
  mjfCollision func = mjCOLLISIONFUNC[m->geom_type[g1]][m->geom_type[g2]];
  if (!(m->opt.disableflags & mjDSBL_NATIVECCD) &&
      (func == mjc_Convex || func == mjc_BoxBox)) {
    if (fromto) mju_zero(fromto, 6);
    return repaired_distance_ccd(m, d, geom1, geom2, distmax, fromto);
  }
  return mj_geomDistance(m, d, geom1, geom2, distmax, fromto);
}

// The loader checks that this DSO resolves MuJoCo to the already loaded library.
void* v34_linked_mj_version(void) {
  return (void*)mj_version;
}
'''


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_library():
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or ctypes.sizeof(ctypes.c_void_p) != 8
            or mujoco.__version__ != '3.6.0' or mujoco.mj_version() != 3006000
            or mujoco.mj_versionString() != '3.6.0'):
        raise RuntimeError('Unsupported runtime for pinned query-only MuJoCo 3.6.0 repair')
    root = Path(mujoco.__file__).resolve().parent
    library = (root / 'libmujoco.so.3.6.0').resolve(strict=True)
    if _sha256(library) != _LIBRARY_SHA256:
        raise RuntimeError('Unsupported libmujoco build; private query ABI is pinned by SHA256')
    include = root / 'include'
    headers = {str(p.relative_to(include)): _sha256(p) for p in sorted(include.rglob('*.h'))}
    header_hash = hashlib.sha256(
        json.dumps(headers, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    if header_hash != _HEADERS_SHA256:
        raise RuntimeError('Unsupported installed public headers for pinned private query ABI')
    mapped = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and 'libmujoco' in Path(fields[5]).name:
            mapped.add(Path(fields[5]).resolve(strict=True))
    if mapped != {library}:
        raise RuntimeError('Python must already use only the pinned libmujoco shared object')
    return library, include


class _PreparedSweepDistance:
    """Serial callable bound to one model and one planning-only MjData."""

    def __init__(self, model, data, library, temporary, metadata):
        self._model = model
        self._data = data
        self._model_pointer = ctypes.c_void_p(model._address)
        self._data_pointer = ctypes.c_void_p(data._address)
        self._library = library
        self._temporary = temporary
        self._query = library.v34_sweep_geom_distance
        self._query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        self._query.restype = ctypes.c_double
        self.metadata = metadata
        self._closed = False

    def __call__(self, geom1, geom2, distmax, fromto=None):
        if self._closed:
            raise RuntimeError('Sweep distance query used after its scratch lifetime ended')
        geom1, geom2 = operator.index(geom1), operator.index(geom2)
        if not (0 <= geom1 < self._model.ngeom and 0 <= geom2 < self._model.ngeom):
            raise ValueError('Sweep distance geom index outside the bound model')
        pointer = None
        if fromto is not None:
            if (not isinstance(fromto, np.ndarray) or fromto.dtype != np.dtype(np.float64)
                    or fromto.shape != (6,) or not fromto.flags.c_contiguous
                    or not fromto.flags.aligned or not fromto.flags.writeable):
                raise ValueError('fromto must be a writable aligned contiguous float64 array of shape (6,)')
            pointer = fromto.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self._query(self._model_pointer, self._data_pointer,
                           geom1, geom2, float(distmax), pointer)

    def close(self):
        self._closed = True
        self._temporary.cleanup()

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Cannot reopen a closed sweep distance query')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def prepare_sweep_distance(model, planning_data, *, scratch_parent):
    """Load frozen query bytes inside the unchanged native execution fence.

    The original query C source is retained above for reproducible external
    builds. Runtime only verifies and loads its immutable artifact; it never
    starts a compiler or relaxes the original subprocess/read/write policy.
    The caller must supply isolated planning data and an owned output directory.
    """
    from v34_smoothtubereturn_distance_binary import BINARY_BASE64, BINARY_SHA256, BUILD

    if not isinstance(model, mujoco.MjModel) or not isinstance(planning_data, mujoco.MjData):
        raise TypeError('Sweep query requires a MuJoCo model and planning MjData')
    if planning_data.model._address != model._address:
        raise ValueError('Planning MjData must belong to the bound model')
    library, _ = _runtime_library()
    source_hash = hashlib.sha256(_C_SOURCE.encode('utf-8')).hexdigest()
    if (BUILD['embedded_c_sha256'] != source_hash
            or BUILD['library_sha256'] != _LIBRARY_SHA256
            or BUILD['installed_headers_manifest_sha256'] != _HEADERS_SHA256
            or BUILD['compiled_query_sha256'] != BINARY_SHA256):
        raise RuntimeError('Frozen native query artifact does not match its source/runtime')
    binary = base64.b64decode(BINARY_BASE64, validate=True)
    if hashlib.sha256(binary).hexdigest() != BINARY_SHA256:
        raise RuntimeError('Frozen native query artifact SHA256 mismatch')
    parent = Path(scratch_parent).resolve(strict=True)
    if not parent.is_dir() or parent.stat().st_uid != os.getuid():
        raise RuntimeError('Sweep query scratch parent must be an existing owned directory')
    temporary = tempfile.TemporaryDirectory(prefix='native_distance_06-', dir=parent)
    try:
        scratch = Path(temporary.name)
        shared = scratch / 'native_distance_06.so'
        shared.write_bytes(binary)
        native = ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_NOW)
        shim = ctypes.CDLL(str(shared), mode=os.RTLD_LOCAL | os.RTLD_NOW)
        shim.v34_linked_mj_version.argtypes = []
        shim.v34_linked_mj_version.restype = ctypes.c_void_p
        if shim.v34_linked_mj_version() != ctypes.cast(native.mj_version, ctypes.c_void_p).value:
            raise RuntimeError('Query shim is not linked to the pinned loaded MuJoCo library')
        metadata = dict(
            repair_id=_REPAIR_ID,
            scope='Query-only; one isolated planning MjData; no live simulation patch',
            repair='Initialize mjCCDConfig.context/alloc/free as native mjc_penetration does',
            implementation_status='Loaded prebuilt query; numerical equivalence is not inferred',
            mujoco_version='3.6.0', library_path=str(library), library_sha256=_LIBRARY_SHA256,
            installed_headers_manifest_sha256=_HEADERS_SHA256,
            sources={name: dict(url=_SOURCE_ROOT + name, sha256=digest)
                     for name, digest in _SOURCE_SHA256.items()},
            helper_sha256=_sha256(__file__), embedded_c_sha256=source_hash,
            compiled_query_sha256=BINARY_SHA256, prebuilt_query_build=BUILD,
            runtime_compiler_invoked=False, native_subprocess_guard_unchanged=True,
            scratch_directory=str(scratch), scratch_cleanup='Context exit/close removes owned library copy',
            observed_ccd_iterations=int(model.opt.ccd_iterations),
            observed_ccd_tolerance=float(model.opt.ccd_tolerance),
            observed_disableflags=int(model.opt.disableflags), observed_enableflags=int(model.opt.enableflags),
            option_policy='Read current original options on every call; no iteration cap or option writes',
            dispatch='Same native-CCD Convex/BoxBox predicate; all other paths use original mj_geomDistance',
            numerical_limits='Original floating-point GJK/EPA and stopping tolerance, distmax clipping and witness semantics; not exact geometry or a strengthened error bound. Arena exhaustion and native errors remain failures.',
            allocation_scope='Native mj_markStack/mj_stackAllocByte(sizeof(mjtNum))/mj_freeStack on planning data; its stack high-water bookkeeping may change',
            no_query_skipping=True, no_error_fallback=True, no_live_model_mutation=True,
        )
        return _PreparedSweepDistance(model, planning_data, shim, temporary, metadata)
    except BaseException:
        temporary.cleanup()
        raise
