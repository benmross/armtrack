// The Loop rover and its arm, built procedurally from photos, driven live by
// stereo-hand over Server-Sent Events. World is z-up, metres, origin at the
// top of the arm's turret on the chassis. Joint conventions match arm.py fk():
// yaw about z, then shoulder/elbow/wrist pitch as elevation in the arm plane,
// then wrist roll about the gripper's approach axis.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// ---------------------------------------------------------------- spec
let SPEC = {
  shoulder_h: 0.14, upper: 0.52, fore: 0.56, wrist: 0.13, finger: 0.13, ground_z: -0.56,
  limits_deg: [[-170, 170], [-5, 120], [-165, -10], [-115, 115], [-180, 180]],
};
try { SPEC = Object.assign(SPEC, await (await fetch('./spec')).json()); } catch (e) { /* static preview */ }
const G = SPEC.ground_z;

// ---------------------------------------------------------------- renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15161a);
scene.fog = new THREE.Fog(0x15161a, 4.5, 11);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.02, 60);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.minDistance = 0.3;
controls.maxDistance = 9;
controls.maxPolarAngle = Math.PI * 0.495;  // never below the ground

const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x3a3128, 0.55);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff4e6, 3.2);
sun.position.set(2.2, -2.6, 4.0);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
sun.shadow.radius = 3;
Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.5, far: 12 });
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.02;
scene.add(sun);
const rim = new THREE.DirectionalLight(0x9fc4ff, 0.6);
rim.position.set(-3, 3, 2);
scene.add(rim);

// ---------------------------------------------------------------- materials
const M = {
  alu: new THREE.MeshStandardMaterial({ color: 0xc8ccd2, metalness: 0.9, roughness: 0.32 }),
  aluBrushed: new THREE.MeshStandardMaterial({ color: 0xb9bdc4, metalness: 0.85, roughness: 0.45 }),
  aluDark: new THREE.MeshStandardMaterial({ color: 0x8e939b, metalness: 0.85, roughness: 0.4 }),
  black: new THREE.MeshStandardMaterial({ color: 0x1a1b1e, metalness: 0.35, roughness: 0.55 }),
  blackGloss: new THREE.MeshStandardMaterial({ color: 0x111214, metalness: 0.5, roughness: 0.25 }),
  white: new THREE.MeshStandardMaterial({ color: 0xeeeeec, metalness: 0.0, roughness: 0.55 }),
  red: new THREE.MeshStandardMaterial({ color: 0xd3232c, metalness: 0.0, roughness: 0.62 }),
  redDark: new THREE.MeshStandardMaterial({ color: 0xa8171f, metalness: 0.0, roughness: 0.7 }),
  chrome: new THREE.MeshStandardMaterial({ color: 0xf2f2f2, metalness: 1.0, roughness: 0.12 }),
  rubber: new THREE.MeshStandardMaterial({ color: 0x232325, metalness: 0.0, roughness: 0.9 }),
  copper: new THREE.MeshStandardMaterial({ color: 0xc9773b, metalness: 0.9, roughness: 0.35 }),
  lens: new THREE.MeshStandardMaterial({ color: 0x0b1a2a, metalness: 0.9, roughness: 0.05 }),
  pcb: new THREE.MeshStandardMaterial({ color: 0x1f5f3a, metalness: 0.2, roughness: 0.6 }),
  cable: new THREE.MeshStandardMaterial({ color: 0x151515, metalness: 0.1, roughness: 0.75 }),
};

// ---------------------------------------------------------------- helpers
function mesh(geo, mat, parent, pos, rot) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  if (pos) m.position.set(...pos);
  if (rot) m.rotation.set(...rot);
  if (parent) parent.add(m);
  return m;
}
const box = (x, y, z, mat, parent, pos, rot) => mesh(new THREE.BoxGeometry(x, y, z), mat, parent, pos, rot);
// cylinders along a chosen axis
const cylX = (r, len, mat, parent, pos, seg = 28) => mesh(new THREE.CylinderGeometry(r, r, len, seg), mat, parent, pos, [0, 0, Math.PI / 2]);
const cylY = (r, len, mat, parent, pos, seg = 28) => mesh(new THREE.CylinderGeometry(r, r, len, seg), mat, parent, pos);
const cylZ = (r, len, mat, parent, pos, seg = 28) => mesh(new THREE.CylinderGeometry(r, r, len, seg), mat, parent, pos, [Math.PI / 2, 0, 0]);
const Y = new THREE.Vector3(0, 1, 0);

// a cylinder (axis along geometry y, unit length) stretched between two points
function strut(r, mat, parent, seg = 18) {
  const m = mesh(new THREE.CylinderGeometry(r, r, 1, seg), mat, parent);
  m.userData.set = (a, b) => {
    const d = new THREE.Vector3().subVectors(b, a);
    const len = d.length();
    m.position.copy(a).addScaledVector(d, 0.5);
    m.scale.set(1, Math.max(len, 1e-4), 1);
    m.quaternion.setFromUnitVectors(Y, d.normalize());
  };
  return m;
}
// a square aluminium beam between two points
function beam(a, b, w, h, mat, parent) {
  const d = new THREE.Vector3().subVectors(b, a);
  const m = box(w, d.length(), h, mat, parent);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(Y, d.clone().normalize());
  return m;
}
// a rounded slot plate (two semicircles joined) in the x-z plane, extruded along y
function slotPlate(len, w, thick, mat, parent, pos, holeR = 0) {
  const s = new THREE.Shape();
  const r = w / 2;
  s.absarc(0, 0, r, Math.PI / 2, Math.PI * 1.5, false);
  s.absarc(len, 0, r, -Math.PI / 2, Math.PI / 2, false);
  if (holeR > 0) {
    for (const cx of [0, len]) { const h = new THREE.Path(); h.absarc(cx, 0, holeR, 0, Math.PI * 2, true); s.holes.push(h); }
  }
  const g = new THREE.ExtrudeGeometry(s, { depth: thick, bevelEnabled: true, bevelThickness: 0.0015, bevelSize: 0.0015, bevelSegments: 2, curveSegments: 20 });
  g.translate(0, 0, -thick / 2);
  g.rotateX(Math.PI / 2);  // shape x-y -> world x-z, extrusion along y
  return mesh(g, mat, parent, pos);
}
// rivets in a line, as one instanced mesh
const rivetGeo = new THREE.SphereGeometry(0.0045, 8, 6);
function rivets(parent, pts) {
  const im = new THREE.InstancedMesh(rivetGeo, M.aluDark, pts.length);
  const m = new THREE.Matrix4();
  pts.forEach((p, i) => im.setMatrixAt(i, m.makeTranslation(p[0], p[1], p[2])));
  im.castShadow = true;
  parent.add(im);
}
function cable(parent, pts, r = 0.006, mat = M.cable) {
  const curve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(...p)));
  return mesh(new THREE.TubeGeometry(curve, 48, r, 8, false), mat, parent);
}

// ---------------------------------------------------------------- ground
{
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#4a453f'; g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 22000; i++) {
    const v = 45 + Math.random() * 55;
    g.fillStyle = `rgb(${v + 12},${v + 4},${v - 6})`;
    const s = Math.random() * 2.6 + 0.6;
    g.fillRect(Math.random() * 512, Math.random() * 512, s, s);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(14, 14);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const ground = mesh(new THREE.PlaneGeometry(24, 24), new THREE.MeshStandardMaterial({ map: tex, roughness: 1.0, envMapIntensity: 0.2 }), scene, [0, 0, G]);
  ground.castShadow = false;
  const grid = new THREE.GridHelper(6, 24, 0x3b3833, 0x3b3833);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = G + 0.001;
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  scene.add(grid);
}

// ---------------------------------------------------------------- rover body
const rover = new THREE.Group();
scene.add(rover);
const CH = { x0: -0.50, x1: 0.22, y: 0.21, z0: -0.22, z1: 0.0 };
{
  const L = CH.x1 - CH.x0, cx = (CH.x0 + CH.x1) / 2, H = CH.z1 - CH.z0, cz = (CH.z0 + CH.z1) / 2;
  box(L, CH.y * 2, H, M.aluBrushed, rover, [cx, 0, cz]);
  // side panels, slightly proud, with seams and rivets like the photo
  for (const s of [-1, 1]) {
    const panels = [[CH.x0 + 0.01, -0.30], [-0.30 + 0.004, -0.06], [-0.06 + 0.004, CH.x1 - 0.01]];
    panels.forEach(([a, b], i) => {
      box(b - a - 0.004, 0.004, H - 0.02, i === 1 ? M.alu : M.aluBrushed, rover, [(a + b) / 2, s * (CH.y + 0.002), cz]);
      const pts = [];
      for (let x = a + 0.015; x < b - 0.01; x += 0.035) { pts.push([x, s * (CH.y + 0.0045), CH.z1 - 0.022]); pts.push([x, s * (CH.y + 0.0045), CH.z0 + 0.022]); }
      for (let z = CH.z0 + 0.05; z < CH.z1 - 0.04; z += 0.04) { pts.push([a + 0.012, s * (CH.y + 0.0045), z]); pts.push([b - 0.012, s * (CH.y + 0.0045), z]); }
      rivets(rover, pts);
    });
  }
  // front panel with the drive camera in a white printed mount
  box(0.004, CH.y * 2 - 0.02, H - 0.02, M.alu, rover, [CH.x1 + 0.002, 0, cz]);
  const camMount = box(0.02, 0.07, 0.06, M.white, rover, [CH.x1 + 0.012, 0.05, cz + 0.01]);
  box(0.016, 0.034, 0.034, M.black, rover, [CH.x1 + 0.028, 0.05, cz + 0.01]);
  cylX(0.009, 0.01, M.lens, rover, [CH.x1 + 0.04, 0.05, cz + 0.01]);
  // rear panel with connectors and a status light
  box(0.004, CH.y * 2 - 0.02, H - 0.02, M.alu, rover, [CH.x0 - 0.002, 0, cz]);
  for (let i = 0; i < 3; i++) cylX(0.008, 0.016, M.black, rover, [CH.x0 - 0.01, -0.12 + i * 0.05, cz - 0.03]);
  const estop = cylZ(0.022, 0.02, new THREE.MeshStandardMaterial({ color: 0xe0242b, roughness: 0.4 }), rover, [CH.x0 + 0.06, -0.14, CH.z1 + 0.012]);
  cylZ(0.03, 0.006, new THREE.MeshStandardMaterial({ color: 0xf2c300, roughness: 0.5 }), rover, [CH.x0 + 0.06, -0.14, CH.z1 + 0.003]);
  // top deck plate and an electronics bay with a board, behind the turret
  box(L - 0.02, CH.y * 2 - 0.02, 0.006, M.alu, rover, [cx, 0, CH.z1 + 0.003]);
  box(0.16, 0.2, 0.004, M.pcb, rover, [-0.30, 0.0, CH.z1 + 0.008]);
  for (let i = 0; i < 4; i++) box(0.02, 0.014, 0.012, M.black, rover, [-0.36 + i * 0.04, 0.06, CH.z1 + 0.016]);
  // the comms mast at the back left: black pole, camera head and antennas
  const mx = -0.46, my = 0.17;
  cylZ(0.016, 1.0, M.black, rover, [mx, my, 0.5]);
  cylZ(0.024, 0.04, M.aluDark, rover, [mx, my, 0.02]);
  box(0.06, 0.08, 0.05, M.black, rover, [mx + 0.01, my, 1.02]);
  cylX(0.014, 0.02, M.lens, rover, [mx + 0.05, my - 0.015, 1.02]);
  cylX(0.014, 0.02, M.lens, rover, [mx + 0.05, my + 0.015, 1.02]);
  cylZ(0.004, 0.32, M.black, rover, [mx - 0.02, my + 0.03, 1.2]);
  cylZ(0.004, 0.22, M.black, rover, [mx - 0.02, my - 0.03, 1.15]);
  cable(rover, [[mx + 0.02, my - 0.01, 0.95], [mx + 0.035, my - 0.02, 0.6], [mx + 0.03, my - 0.03, 0.2], [mx + 0.08, my - 0.06, 0.01]], 0.005);
}

// ---------------------------------------------------------------- suspension and wheels
const WR = 0.12, WW = 0.095, WY = 0.43, HUBZ = G + WR;
function wheel(x, s) {
  const w = new THREE.Group();
  w.position.set(x, s * WY, HUBZ);
  rover.add(w);
  cylY(WR, WW, M.red, w, [0, 0, 0], 40);
  const lug = new THREE.BoxGeometry(0.026, WW * 0.98, 0.016);
  for (let i = 0; i < 22; i++) {
    const a = (i / 22) * Math.PI * 2;
    const l = mesh(lug, M.redDark, w, [Math.cos(a) * (WR + 0.004), 0, Math.sin(a) * (WR + 0.004)], [0, -a, 0]);
    l.rotation.y = -a + Math.PI / 2;
  }
  // hub: aluminium disc with lightening holes on the outside face, black cap
  const face = s * (WW / 2 + 0.002);
  cylY(0.072, 0.008, M.aluBrushed, w, [0, face, 0], 32);
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    cylY(0.012, 0.0085, M.black, w, [Math.cos(a) * 0.045, face, Math.sin(a) * 0.045], 14);
  }
  cylY(0.022, 0.02, M.black, w, [0, face + s * 0.006, 0]);
  // drive motor inboard
  cylY(0.032, 0.09, M.black, w, [0, -s * (WW / 2 + 0.05), 0]);
  cylY(0.034, 0.012, M.aluDark, w, [0, -s * (WW / 2 + 0.096), 0]);
  return w;
}
function bracket(x, s, topZ, steer) {
  // truss plate in the x-z plane between the hub and the leg above it
  const sh = new THREE.Shape();
  sh.moveTo(-0.06, 0); sh.lineTo(0.06, 0); sh.lineTo(0.035, topZ - HUBZ); sh.lineTo(-0.035, topZ - HUBZ); sh.closePath();
  const hole = new THREE.Path();
  const hz = (topZ - HUBZ);
  hole.moveTo(-0.03, 0.03); hole.lineTo(0.03, 0.03); hole.lineTo(0, hz - 0.03); hole.closePath();
  sh.holes.push(hole);
  const g = new THREE.ExtrudeGeometry(sh, { depth: 0.006, bevelEnabled: false });
  g.rotateX(Math.PI / 2);
  const yIn = s * (WY - WW / 2 - 0.11);
  for (const dy of [0, s * 0.03]) mesh(g, M.alu, rover, [x, yIn + dy, HUBZ]);
  if (steer) {  // steering servo block on the corner wheels
    box(0.06, 0.05, 0.04, M.white, rover, [x, yIn + s * 0.015, topZ + 0.02]);
    cylZ(0.012, 0.03, M.black, rover, [x, yIn + s * 0.015, topZ + 0.05]);
  }
  return new THREE.Vector3(x, yIn + s * 0.015, topZ);
}
for (const s of [-1, 1]) {
  const piv = new THREE.Vector3(-0.14, s * (CH.y + 0.03), -0.12);
  cylY(0.03, 0.04, M.black, rover, [piv.x, piv.y, piv.z]);
  // rocker: front leg to the front corner wheel, rear leg to the bogie pivot
  wheel(0.36, s); const fTop = bracket(0.36, s, -0.30, true);
  beam(piv, fTop, 0.034, 0.034, M.alu, rover);
  const bog = new THREE.Vector3(-0.42, s * (WY - WW / 2 - 0.095), -0.33);
  beam(piv, bog, 0.034, 0.034, M.alu, rover);
  cylY(0.025, 0.05, M.black, rover, [bog.x, bog.y, bog.z]);
  wheel(-0.18, s); const mTop = bracket(-0.18, s, -0.36, false);
  wheel(-0.68, s); const rTop = bracket(-0.68, s, -0.30, true);
  beam(bog, mTop, 0.03, 0.03, M.aluBrushed, rover);
  beam(bog, rTop, 0.03, 0.03, M.aluBrushed, rover);
  // wiring to the wheel motors
  cable(rover, [[piv.x, piv.y, piv.z + 0.03], [0.1, s * 0.3, -0.2], [0.33, s * 0.33, -0.34], [0.36, s * (WY - WW / 2 - 0.06), HUBZ + 0.02]], 0.005);
  cable(rover, [[piv.x - 0.02, piv.y, piv.z + 0.02], [-0.32, s * 0.32, -0.26], [-0.5, s * 0.33, -0.36], [-0.66, s * (WY - WW / 2 - 0.06), HUBZ + 0.02]], 0.005);
}

// ---------------------------------------------------------------- the arm
const arm = new THREE.Group();        // fixed to the chassis at the turret
scene.add(arm);
cylZ(0.11, 0.022, M.black, arm, [0, 0, 0.011], 40);       // slewing bearing
cylZ(0.115, 0.006, M.aluDark, arm, [0, 0, 0.003], 40);
const yaw = new THREE.Group(); yaw.position.z = 0.022; arm.add(yaw);
cylZ(0.095, 0.02, M.alu, yaw, [0, 0, 0.01], 40);          // turret plate
for (const s of [-1, 1]) {
  // shoulder clevis plates
  box(0.17, 0.012, 0.15, M.alu, yaw, [-0.01, s * 0.07, 0.085]);
  rivets(yaw, [[-0.08, s * 0.077, 0.03], [0.06, s * 0.077, 0.03], [-0.08, s * 0.077, 0.14], [0.06, s * 0.077, 0.14]]);
  // big shoulder motors with gearboxes, white bodies and black caps, as in the photo
  cylY(0.046, 0.11, M.white, yaw, [0, s * 0.145, SPEC.shoulder_h - 0.022]);
  cylY(0.048, 0.016, M.black, yaw, [0, s * 0.205, SPEC.shoulder_h - 0.022]);
  cylY(0.036, 0.03, M.black, yaw, [0, s * 0.09, SPEC.shoulder_h - 0.022]);
  for (let i = 0; i < 4; i++) cylY(0.0485, 0.003, M.black, yaw, [0, s * (0.11 + i * 0.022), SPEC.shoulder_h - 0.022]);
  // actuator anchor lugs at the back of the turret
  box(0.04, 0.014, 0.05, M.aluDark, yaw, [-0.15, s * 0.045, 0.04]);
}
cable(yaw, [[-0.05, 0.0, 0.03], [-0.12, -0.06, 0.06], [-0.1, -0.1, 0.0], [-0.05, -0.12, -0.02]], 0.006);
const SH = SPEC.shoulder_h - 0.022;   // shoulder axis height inside the yaw group

const shoulder = new THREE.Group(); shoulder.position.z = SH; yaw.add(shoulder);
cylY(0.05, 0.13, M.black, shoulder, [0, 0, 0]);
cylY(0.052, 0.012, M.aluDark, shoulder, [0, 0.062, 0]);
cylY(0.052, 0.012, M.aluDark, shoulder, [0, -0.062, 0]);
cylX(0.029, SPEC.upper - 0.05, M.alu, shoulder, [SPEC.upper / 2, 0, 0], 32);
for (const f of [0.22, 0.42, 0.62]) {  // white printed clamps along the upper arm
  cylX(0.036, 0.03, M.white, shoulder, [SPEC.upper * f, 0, 0]);
  box(0.03, 0.10, 0.018, M.white, shoulder, [SPEC.upper * f, 0, -0.03]);
}
box(0.05, 0.014, 0.03, M.blue = new THREE.MeshStandardMaterial({ color: 0x2f6fdb, roughness: 0.5 }), shoulder, [SPEC.upper * 0.5, 0.03, 0]);
const ACT_ON = [SPEC.upper * 0.62, 0.045, -0.04];  // where the actuators meet the upper arm
cable(shoulder, [[0.05, -0.03, 0.03], [0.2, -0.035, 0.035], [0.4, -0.03, 0.03], [SPEC.upper - 0.05, -0.04, 0.04]], 0.005);

const elbow = new THREE.Group(); elbow.position.x = SPEC.upper; shoulder.add(elbow);
for (const s of [-1, 1]) slotPlate(0.12, 0.09, 0.008, M.black, elbow, [-0.03, s * 0.05, 0], 0.016);
cylY(0.04, 0.115, M.blackGloss, elbow, [0, 0, 0]);
cylY(0.036, 0.07, M.white, elbow, [0, 0.095, 0]);   // elbow motor
cylY(0.038, 0.01, M.black, elbow, [0, 0.135, 0]);
cylX(0.026, SPEC.fore - 0.07, M.alu, elbow, [SPEC.fore / 2 - 0.005, 0, 0], 32);
cylX(0.033, 0.035, M.white, elbow, [SPEC.fore * 0.38, 0, 0]);
box(0.04, 0.012, 0.07, M.black, elbow, [SPEC.fore * 0.38, 0, 0.05]);   // the black tab on the forearm clamp
cable(elbow, [[0.0, -0.045, 0.03], [0.2, -0.03, 0.03], [SPEC.fore - 0.06, -0.035, 0.035]], 0.005);

const wrist = new THREE.Group(); wrist.position.x = SPEC.fore; elbow.add(wrist);
for (const s of [-1, 1]) {
  slotPlate(0.085, 0.11, 0.008, M.aluBrushed, wrist, [-0.03, s * 0.058, 0], 0.022);
  cylY(0.028, 0.006, M.black, wrist, [0, s * 0.066, 0]);
}
cylY(0.042, 0.1, M.black, wrist, [0, 0, 0]);
cylY(0.04, 0.05, M.white, wrist, [0, -0.09, 0]);   // wrist pitch motor
cylX(0.046, 0.06, M.white, wrist, [0.06, 0, 0]);
cylX(0.048, 0.008, M.black, wrist, [0.09, 0, 0]);
const roll = new THREE.Group(); roll.position.x = 0.09; wrist.add(roll);
cylX(0.04, 0.04, M.black, roll, [0.02, 0, 0]);
box(0.012, 0.12, 0.05, M.aluDark, roll, [SPEC.wrist - 0.09 - 0.006, 0, 0]);   // palm plate
box(0.03, 0.1, 0.012, M.black, roll, [SPEC.wrist - 0.09 - 0.02, 0, -0.03]);    // finger rail
box(0.028, 0.03, 0.026, M.black, roll, [0.02, 0.0, 0.055]);                     // wrist camera
cylX(0.008, 0.008, M.lens, roll, [0.036, 0.0, 0.055]);
const fingers = [];
for (const s of [-1, 1]) {
  const f = new THREE.Group(); f.position.x = SPEC.wrist - 0.09; roll.add(f);
  box(0.024, 0.022, 0.04, M.black, f, [0.012, 0, 0]);
  const fin = box(SPEC.finger - 0.02, 0.014, 0.034, M.red, f, [(SPEC.finger - 0.02) / 2 + 0.02, 0, 0]);
  box(SPEC.finger - 0.045, 0.004, 0.026, M.white, f, [(SPEC.finger - 0.045) / 2 + 0.04, -s * 0.009, 0]);   // grip pad
  box(0.018, 0.014, 0.022, M.red, f, [SPEC.finger - 0.003, -s * 0.004, 0]);                                 // fingertip
  f.userData.s = s;
  fingers.push(f);
}

// linear actuators: black body plus chrome rod, from the turret anchor to the upper arm
const acts = [];
for (const s of [-1, 1]) acts.push({ s, body: strut(0.019, M.black, arm), rod: strut(0.009, M.chrome, arm), cap: strut(0.022, M.blackGloss, arm) });
const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();
function placeActuators() {
  for (const a of acts) {
    tmpA.set(-0.15, a.s * 0.045, 0.04); yaw.localToWorld(tmpA); arm.worldToLocal(tmpA);
    tmpB.set(ACT_ON[0], a.s * ACT_ON[1], ACT_ON[2]); shoulder.localToWorld(tmpB); arm.worldToLocal(tmpB);
    const bodyEnd = tmpA.clone().lerp(tmpB, 0.55);
    a.body.userData.set(tmpA, bodyEnd);
    a.cap.userData.set(bodyEnd.clone().lerp(tmpA, 0.04), bodyEnd);
    a.rod.userData.set(bodyEnd, tmpB);
  }
}

// ---------------------------------------------------------------- target, trail
const ghost = mesh(new THREE.SphereGeometry(0.028, 24, 16), new THREE.MeshStandardMaterial({ color: 0x3ccf6e, transparent: true, opacity: 0.45, emissive: 0x1a6d36 }), scene);
ghost.castShadow = false;
const ghostRing = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.04, 32), new THREE.MeshBasicMaterial({ color: 0x3ccf6e, transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
scene.add(ghostRing);
const lagLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({ color: 0x3ccf6e, dashSize: 0.02, gapSize: 0.015 }));
scene.add(lagLine);
const TRAIL_N = 400;
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
trailGeo.setDrawRange(0, 0);
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xf2a93b, transparent: true, opacity: 0.8 }));
trail.visible = false;
scene.add(trail);
let trailCount = 0;

// ---------------------------------------------------------------- pose
const Q = [0, 1.3, -2.1, 0.4, 0];   // shown pose, eased toward the received one
let Qrx = Q.slice(), grip = 1, gripRx = 1;
function applyPose() {
  yaw.rotation.z = Q[0];
  shoulder.rotation.y = -Q[1];
  elbow.rotation.y = -Q[2];
  wrist.rotation.y = -Q[3];
  roll.rotation.x = Q[4];
  const spread = 0.012 + 0.042 * grip;
  for (const f of fingers) f.position.y = f.userData.s * spread;
  arm.updateMatrixWorld(true);
  placeActuators();
}
function tipWorld() {
  const p = new THREE.Vector3(SPEC.wrist - 0.09 + SPEC.finger, 0, 0);
  return roll.localToWorld(p);
}

// ---------------------------------------------------------------- HUD
const NAMES = ['Base yaw', 'Shoulder', 'Elbow', 'Wrist pitch', 'Wrist roll', 'Gripper'];
const tbl = document.getElementById('joints');
tbl.innerHTML = NAMES.map((n, i) => `<tr><td class="name">${n}</td><td><div class="bar"><b id="s${i}"></b><i id="b${i}"></i></div></td><td class="val" id="v${i}"></td></tr>`).join('');
const $ = id => document.getElementById(id);
function hud(st) {
  for (let i = 0; i < 5; i++) {
    const deg = Q[i] * 180 / Math.PI, [lo, hi] = SPEC.limits_deg[i];
    $('b' + i).style.left = (100 * (deg - lo) / (hi - lo)).toFixed(1) + '%';
    $('s' + i).style.width = (100 * Math.min(1, (st.speed_frac || [])[i] || 0)).toFixed(0) + '%';
    $('v' + i).textContent = deg.toFixed(1) + '°';
  }
  $('b5').style.left = (100 * grip).toFixed(0) + '%';
  $('s5').style.width = '0%';
  $('v5').textContent = grip > 0.95 ? 'open' : grip < 0.05 ? 'closed' : (100 * grip).toFixed(0) + '%';
  const e = $('engaged');
  e.textContent = st.engaged ? 'ENGAGED' : 'HOLDING';
  e.className = 'pill ' + (st.engaged ? 'on' : 'off');
  $('reach').style.display = st.reachable === false ? 'inline-block' : 'none';
  $('status').textContent = st.status || '';
  const lag = st.lag_m || 0;
  $('lag').textContent = lag > 0.01 ? `Arm catching up: ${(lag * 100).toFixed(0)} cm behind the command (amber bars are joint speed)` : 'Arm on target';
}

// ---------------------------------------------------------------- views
const VIEWS = {
  iso: { pos: [1.9, -2.1, 1.2], tgt: [0.15, 0, -0.05] },
  side: { pos: [0.2, -3.0, 0.15], tgt: [0.2, 0, -0.05] },
  front: { pos: [3.0, 0, 0.35], tgt: [0.1, 0, -0.05] },
  top: { pos: [0.15, -0.001, 3.4], tgt: [0.15, 0, -0.1] },
};
let fly = null;
function setView(name, instant = false) {
  const v = VIEWS[name];
  fly = { p0: camera.position.clone(), t0: controls.target.clone(), p1: new THREE.Vector3(...v.pos), t1: new THREE.Vector3(...v.tgt), k: instant ? 1 : 0 };
}
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => { follow = false; $('follow').classList.remove('active'); setView(b.dataset.view); });
let follow = false;
$('follow').onclick = () => { follow = !follow; $('follow').classList.toggle('active', follow); };
$('trail').onclick = () => { trail.visible = !trail.visible; $('trail').classList.toggle('active', trail.visible); trailCount = 0; };
camera.position.set(...VIEWS.iso.pos);
controls.target.set(...VIEWS.iso.tgt);

// ---------------------------------------------------------------- data
let state = { engaged: false, status: '' }, lastRx = 0;
const params = new URLSearchParams(location.search);
if (params.has('preview')) {
  // no tracker: sweep the joints so the model can be inspected on its own
  lastRx = Infinity;
} else {
  const es = new EventSource('./stream');
  es.onmessage = ev => {
    state = JSON.parse(ev.data);
    lastRx = performance.now();
    if (state.q) Qrx = state.q;
    if (state.gripper_open != null) gripRx = state.gripper_open;
  };
}
if (params.has('view')) setView(params.get('view'), true);

// ---------------------------------------------------------------- loop
const timer = new THREE.Timer();
function frame() {
  timer.update();
  const dt = Math.min(timer.getDelta(), 0.1), t = timer.getElapsed();
  if (params.has('preview')) {
    Qrx = [0.6 * Math.sin(t * 0.4), 0.9 + 0.3 * Math.sin(t * 0.5), -1.6 + 0.4 * Math.sin(t * 0.7), -0.2 + 0.5 * Math.sin(t * 0.9), Math.sin(t * 0.6)];
    gripRx = 0.5 + 0.5 * Math.sin(t * 1.3);
    state = { engaged: true, status: 'preview: no tracker connected', reachable: true, speed_frac: [0.3, 0.2, 0.4, 0.5, 0.6] };
  }
  // ease toward the received pose so network jitter does not show
  const k = 1 - Math.exp(-dt * 25);
  for (let i = 0; i < 5; i++) {
    let d = Qrx[i] - Q[i];
    if (i === 0 || i === 4) d = Math.atan2(Math.sin(d), Math.cos(d));
    Q[i] += d * k;
  }
  grip += (gripRx - grip) * k;
  applyPose();

  const tip = tipWorld();
  // the drawn tip must sit where Python's fk() says; exposed for the render test
  if (state.tip_m && Math.abs(Q[0] - Qrx[0]) + Math.abs(Q[1] - Qrx[1]) + Math.abs(Q[2] - Qrx[2]) + Math.abs(Q[3] - Qrx[3]) < 1e-4)
    window.__tipErr = tip.distanceTo(new THREE.Vector3(...state.tip_m));
  const tg = state.target_tip_m ? new THREE.Vector3(...state.target_tip_m) : tip.clone();
  ghost.position.copy(tg);
  ghostRing.position.copy(tg); ghostRing.lookAt(camera.position);
  const col = state.reachable === false ? 0xf2a93b : 0x3ccf6e;
  ghost.material.color.setHex(col); ghostRing.material.color.setHex(col); lagLine.material.color.setHex(col);
  const far = tip.distanceTo(tg) > 0.01;
  ghost.visible = ghostRing.visible = lagLine.visible = far || !!state.engaged;
  lagLine.geometry.setFromPoints([tip, tg]); lagLine.computeLineDistances();

  if (trail.visible) {
    const arr = trailGeo.attributes.position.array;
    if (trailCount < TRAIL_N) trailCount++; else arr.copyWithin(0, 3);
    arr.set([tip.x, tip.y, tip.z], (trailCount - 1) * 3);
    trailGeo.attributes.position.needsUpdate = true;
    trailGeo.setDrawRange(0, trailCount);
  }

  if (fly) {
    fly.k = Math.min(1, fly.k + dt * 1.8);
    const e = fly.k < 1 ? 1 - Math.pow(1 - fly.k, 3) : 1;
    camera.position.lerpVectors(fly.p0, fly.p1, e);
    controls.target.lerpVectors(fly.t0, fly.t1, e);
    if (fly.k >= 1) fly = null;
  } else if (follow) {
    const d = tip.clone().sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 4));
    controls.target.add(d); camera.position.add(d);
  }
  controls.update();
  hud(state);
  $('wait').style.display = performance.now() - lastRx > 1500 ? 'block' : 'none';
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight);
});
window.__ready = true;
frame();
