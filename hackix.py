#!/usr/bin/env python
# Written by Paul Brook and ???
# Released under the GNU GPL v3.

from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import random
import math
import numpy
import numpy.linalg

def mat_to_numpy(mat):
    return numpy.array([mat[0::4], mat[1::4], mat[2::4], mat[3::4]])

def mat_from_numpy(a):
    res=[]
    for col in range(0, 4):
	for row in range(0, 4):
	    res.append(a[row][col])
    return res;

def resize((width, height)):
    if height==0:
        height=1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnable(GL_CULL_FACE)

class Vector(object):
    def __init__(self, x, y, z):
	self.x = x
	self.y = y
	self.z = z
    def __sub__(self, val):
	return Vector(self.x - val.x, self.y - val.y, self.z - val.z)
    def dot_product(self, val):
	return self.x * val.x + self.y * val.y + self.z * val.z
    def magnitude(self):
	return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    def normalize(self):
	n = 1.0 / self.magnitude()
	return Vector(self.x * n, self.y * n, self.z * n)

class Point(Vector):
    def __init__(self, x, y, z):
	super(self.__class__, self).__init__(x, y, z)

def vec_fracpoint(a, b, frac):
    x = (b.x - a.x) * frac + a.x
    y = (b.y - a.y) * frac + a.y
    z = (b.z - a.z) * frac + a.z
    return Vector(x, y, z)

class SplitEdge:
    def __init__(self, point, edge):
	self.point = point
	self.edge = edge

# From point A to point B
class Edge(object):
    def __init__(self, a, b):
	self.a = a
	self.b = b
	self.split_edge = None
    def fracpoint(self, frac):
	return vec_fracpoint(self.a, self.b, frac)
    def vector(self):
	return self.b - self.a

class S:
    def __init__(self):
	self.index = 0;
    def bar(self):
	self.index += 1
	return self.index
sfoo = S()

# 3 edges and a color  Edges must be in order
class Face(object):
    def __init__(self, a, b, c, color, oa, ob, oc):
	self.a = a
	self.b = b
	self.c = c
	self.own_a = oa
	self.own_b = ob
	self.own_c = oc
	self.color = color
	self.index = sfoo.bar()
    def get_points(self):
	pa = self.a.a if self.own_a else self.a.b
	pb = self.b.a if self.own_b else self.b.b
	pc = self.c.a if self.own_c else self.c.b
	return (pa, pb, pc)
    def has_edge(self, edge):
	return edge == self.a or edge == self.b or edge == self.c

class Plane(object):
    def __init__(self, p, n):
	self.p = p
	self.n = n
    def behind(self, v):
	return self.n.dot_product(v - self.p) < 0
    def intersect(self, p, p2):
	n = p2 - p
	p0 = self.p - p
	v = self.n.dot_product(p0) / self.n.dot_product(n)
	return Point(p.x + v * n.x, p.y + v * n.y, p.z + v * n.z)

class HackRender(object):
    def update_faces(self, faces):
	self.vertexbuffer=[]
	self.colorbuffer=[]
	self.vertexcount = 0
	for f in faces:
	    for p in f.get_points():
		self.vertexbuffer.append((p.x, p.y, p.z))
		self.colorbuffer.append(f.color)
	    self.vertexcount += 3

    def draw(self, transform, stix_pos):
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
	glLoadIdentity()
	glTranslatef(0.0, 0.0, -5.0)
	glMultMatrixf(transform)
	glVertexPointerf(self.vertexbuffer)
	glColorPointerf(self.colorbuffer)
	glDrawArrays(GL_TRIANGLES, 0, self.vertexcount)

	stix_vertex = []
	stix_color = []
	for (x, y, z) in ((1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)):
	    stix_vertex.append((stix_pos.x + x * 0.1, stix_pos.y + y * 0.1, stix_pos.z + z * 0.1))
	    stix_color.append((0.3, 0.3, 0.3))
	glVertexPointerf(stix_vertex)
	glColorPointerf(stix_color)
	glDrawElementsui(GL_TRIANGLE_STRIP, [0, 1, 2, 3, 4, 5, 0, 1])
	glDrawElementsui(GL_TRIANGLES, [1,5,3]);
	glDrawElementsui(GL_TRIANGLES, [0,2,4]);

def create_cube():
    points=[]
    for x in (1.0, -1.0):
	for y in (1.0, -1.0):
	    for z in (1.0, -1.0):
		points.append(Point(x, y, z))
    edge = []
    for (a, b) in [(0,2), (1,2), (1,3), (1,7), (1,5), (0,5), \
		   (4,5), (4,7), (6,7), (6,3), (6,2), (4,2), \
		   (0,1), (2,3), (3,7), (7,5), (4,0), (6,4)]:
	edge.append(Edge(points[a], points[b]))
    faces=[]
    r = (1.0, 0.0, 0.0)
    g = (0.0, 1.0, 0.0)
    b = (0.0, 0.0, 1.0)
    p = (1.0, 0.0, 1.0)
    y = (1.0, 1.0, 0.0)
    c = (0.0, 1.0, 1.0)
    w = (1.0, 1.0, 1.0)
    for (a, b, c, color) in [(0,1,12,r), (1,-13,2,r), (2,-14,3,g), (3,-15,4,g), \
			     (4,5,-12,b), (5,6,-16,b), (6,15,7,p), (7,8,-17,p), \
			     (8,14,9,c), (9,13,10,c), (10,11,17,y), (11,0,16,y)]:
	ea = edge[a]
	eb = edge[b if b >= 0 else -b]
	ec = edge[c if c >= 0 else -c]
	faces.append(Face(ea, eb, ec, color, True, b < 0, c < 0))
    return faces

def make_poly(edges):
    x = 0.0
    y = 0.0
    z = 0.0
    new_color = (random.random(), random.random(), random.random())
    for e in edges:
	x += e.a.x
	y += e.a.y
	z += e.a.z
    n = len(edges)
    origin = Point(x / n, y / n, z / n)
    # All edges are backwards
    cur_point = edges[0].b
    first_edge = Edge(cur_point, origin)
    prev = first_edge
    while len(edges) > 1:
	for i in range(0, len(edges)):
	    if edges[i].b == cur_point:
		break;
	cur_point = edges[i].a
	new_edge = Edge(cur_point, origin)
	# FIXME do a new color
	yield Face(new_edge, prev, edges[i], new_color, True, False, False)
	prev = new_edge
	del edges[i]
    yield Face(first_edge, prev, edges[0], new_color, True, False, False)

def do_split(faces, plane, stix):
    newfaces = []
    new_poly = []
    for f in faces:
	p = list(f.get_points())
	ba = plane.behind(p[0])
	bb = plane.behind(p[1])
	bc = plane.behind(p[2])
	p = [(p[0], f.own_a), (p[1], f.own_b), (p[2], f.own_c)]
	if ba == bb and bb == bc:
	    if not ba:
		newfaces.append(f)
	    continue
	if (ba == bb): # Point C
	    p = [p[2], p[0], p[1]]
	    eb = f.c
	    ec = f.b
	    opposite_edge = f.a
	    drop_point = bc
	elif bc == ba: # Point B
	    p = [p[1], p[2], p[0]]
	    eb = f.b
	    ec = f.a
	    opposite_edge = f.c
	    drop_point = bb
	elif bb == bc: # Point A
	    eb = f.a
	    ec = f.c
	    opposite_edge = f.b
	    drop_point = ba
	else:
	    raise Exception
	new_split_b = eb.split_edge is None
	if new_split_b:
	    ib = plane.intersect(p[0][0], p[1][0])
	    if drop_point:
		split_edge = Edge(ib, p[1][0])
	    else:
		split_edge = Edge(p[0][0], ib)
	    eb.split_edge = SplitEdge(ib, split_edge)
	new_split_c = ec.split_edge is None
	if new_split_c:
	    ic = plane.intersect(p[0][0], p[2][0])
	    if drop_point:
		split_edge = Edge(p[2][0], ic)
	    else:
		split_edge = Edge(ic, p[0][0])
	    ec.split_edge = SplitEdge(ic, split_edge)
	if drop_point:
	    mid_edge = Edge(eb.split_edge.point, p[2][0])
	    nfb = Face(eb.split_edge.edge, opposite_edge, mid_edge, f.color, new_split_b, p[1][1], False)
	    newfaces.append(nfb)
	    new_edge = Edge(ec.split_edge.point, eb.split_edge.point)
	    nfc = Face(mid_edge, ec.split_edge.edge, new_edge, f.color, True, new_split_c, True);
	    newfaces.append(nfc)
	    if f == stix.current_face:
		stix.check_edges(eb, nfb, ec, nfc, new_edge, nfc)
	else:
	    new_edge = Edge(eb.split_edge.point, ec.split_edge.point)
	    nf = Face(eb.split_edge.edge, new_edge, ec.split_edge.edge, f.color, new_split_b, True, new_split_c)
	    newfaces.append(nf)
	    if f == stix.current_face:
		if stix.to_edge == opposite_edge:
		    # FIXME(again): leaving to_frac is crappy and wrong, but at least it's an answer :-/
		    stix.to_edge = new_edge;
		    stix.current_face = nf;
		else:
		    stix.check_edges(eb, nf, ec, nf, new_edge, nf)
	new_poly.append(new_edge)

    if len(new_poly) > 0:
	newfaces += list(make_poly(new_poly))

    return newfaces

def rotate_x(matrix, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    a = numpy.array([[1.0, 0.0, 0.0, 0.0],
		     [0.0,   c,   s, 0.0],
		     [0.0,  -s,   c, 0.0],
		     [0.0, 0.0, 0.0, 1.0]])
    return numpy.dot(a, matrix)

def rotate_y(matrix, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    a = numpy.array([[  c, 0.0,  -s, 0.0],
		     [0.0, 1.0, 0.0, 0.0],
		     [  s, 0.0,   c, 0.0],
		     [0.0, 0.0, 0.0, 1.0]])
    return numpy.dot(a, matrix)

def rotate_z(matrix, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    a = numpy.array([[  c,   s, 0.0, 0.0],
		     [ -s,   c, 0.0, 0.0],
		     [0.0, 0.0, 1.0, 0.0],
		     [0.0, 0.0, 0.0, 1.0]])
    return numpy.dot(a, matrix)


class Stix(object):
    def __init__(self, face):
	self.from_edge = face.a
	self.from_frac = 0.5
	self.to_edge = face.b
	self.current_face = face
	self.to_frac = 0.5
	self.face_frac = 0.0
	self.speed = 0.5
	self.pos = self.from_edge.fracpoint(self.from_frac)

    # Walk over the surface of the object.
    # This is supposed to walk in a straight line, however there's something
    # wrong with the calculations so we turn a bit when crossing an edge.
    def move(self, faces, delta):
	dist = delta * self.speed;
	while True:
	    from_point = self.from_edge.fracpoint(self.from_frac)
	    to_point = self.to_edge.fracpoint(self.to_frac)
	    face_vec = to_point - from_point
	    face_len = face_vec.magnitude()
	    self.face_frac += dist / face_len
	    if self.face_frac < 1.0:
		break;
	    dist = (self.face_frac - 1.0) * face_len;
	    self.face_frac = 0;
	    new_face = None
	    for f in faces:
		if f == self.current_face:
		    continue
		if f.has_edge(self.to_edge):
		    new_face = f
		    break
	    if new_face is None:
		raise Exception
	    edge_vec = self.to_edge.vector().normalize()
	    stix_dot = edge_vec.dot_product(face_vec.normalize())
	    (pa, pb, pc) = new_face.get_points()
	    if new_face.a == self.to_edge:
		far_point = pc
		edge_l = new_face.c
		edge_r = new_face.b
		ccw = new_face.own_a
	    elif new_face.b == self.to_edge:
		far_point = pa
		edge_l = new_face.a
		edge_r = new_face.c
		ccw = new_face.own_b
	    else: # new_face.c == self.to_edge
		far_point = pb
		edge_l = new_face.b
		edge_r = new_face.a
		ccw = new_face.own_c
	    far_dot = edge_vec.dot_product((far_point - to_point).normalize())
	    self.from_edge = self.to_edge
	    self.from_frac = self.to_frac
	    self.current_face = new_face
	    stix_angle = math.acos(stix_dot)
	    far_angle = math.acos(far_dot)
	    if (stix_angle > far_angle):
		self.to_edge = edge_l if ccw else edge_r
		self.to_frac = (stix_angle - far_angle) / (2 * math.pi - far_angle)
		if self.to_edge.a != far_point:
		    self.to_frac = 1 - self.to_frac
	    else:
		self.to_edge = edge_r if ccw else edge_l
		self.to_frac = stix_angle / far_angle;
		if self.to_edge.b != far_point:
		    self.to_frac = 1 - self.to_frac
	self.pos = vec_fracpoint(from_point, to_point, self.face_frac)
    def fixup_edge(self, old_edge, new_face, split_edge, split_face):
	frac = self.to_frac
	new_edge = old_edge.split_edge.edge
	mult = old_edge.vector().magnitude() / new_edge.vector().magnitude()
	if old_edge.a == new_edge.a:
	    frac *= mult
	elif old_edge.a == new_edge.b:
	    frac = 1 - frac * mult
	elif old_edge.b == new_edge.a:
	    frac = (1 - frac) * mult
	elif old_edge.b == new_edge.b:
	    frac = 1 - ((1 - frac) * mult)
	else:
	    raise Exception
	if (frac >= 0.0) and (frac <= 1.0):
	    self.to_edge = new_edge
	    self.to_frac = frac
	    self.current_face = new_face
	else:
	    # FIXME: leaving to_frac is crappy and wrong, but at least it's an answer :-/
	    self.to_edge = split_edge;
	    self.current_face = split_face;

    def check_edges(self, edge_b, face_b, edge_c, face_c, split_edge, split_face):
	if self.to_edge == edge_b:
	    self.fixup_edge(edge_b, face_b, split_edge, split_face)
	if self.to_edge == edge_c:
	    self.fixup_edge(edge_c, face_c, split_edge, split_face)

    # Compensate for edges that have been split
    def adjust_edges(self):
	def fixup_edge(old_edge, frac):
	    new_edge = old_edge.split_edge.edge
	    mult = old_edge.vector().magnitude() / new_edge.vector().magnitude()
	    if old_edge.a == new_edge.a:
		frac *= mult
	    elif old_edge.a == new_edge.b:
		frac = 1 - frac * mult
	    elif old_edge.b == new_edge.a:
		frac = (1 - frac) * mult
	    elif old_edge.b == new_edge.b:
		frac = 1 - ((1 - frac) * mult)
	    else:
		raise Exception
	    return (new_edge, frac)

	if self.from_edge.split_edge is not None:
	    (self.from_edge, self.from_frac) = fixup_edge(self.from_edge, self.from_frac)
	if self.to_edge.split_edge is not None:
	    (self.to_edge, self.to_frac) = fixup_edge(self.to_edge, self.to_frac)

def matmul3(m, v):
    x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3]
    y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3]
    z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3]
    return Vector(x, y, z)

def main():

    video_flags = OPENGL|DOUBLEBUF
    
    pygame.init()
    pygame.display.set_mode((640,480), video_flags)

    resize((640,480))
    init()

    render = HackRender();
    frames = 0
    faces = create_cube();
    stix = Stix(faces[11])
    render.update_faces(faces)

    transform = numpy.array([[1.0, 0.0, 0.0, 0.0], \
			     [0.0, 1.0, 0.0, 0.0], \
			     [0.0, 0.0, 1.0, 0.0], \
			     [0.0, 0.0, 0.0, 1.0]]);

    dx = 0
    dy = 0
    dz = 0
    last_tick = pygame.time.get_ticks()
    while 1:
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break

	now = pygame.time.get_ticks()
	if event.type == KEYDOWN:
	    if event.key == K_SPACE:
		inv = numpy.linalg.inv(transform)
		origin = matmul3(inv, Vector(0.0, 0.0, 0.0))
		normal = matmul3(inv, Vector(1.0, 0.0, 0.0))
		normal = normal - origin;
		cut = Plane(origin, normal)
		if cut.behind(stix.pos):
		    cut = Plane(origin, Vector(0.0, 0.0, 0.0) - normal)
		faces = do_split(faces, cut, stix)
		render.update_faces(faces)
	    elif event.key == K_LEFT:
		dy = -1.0
	    elif event.key == K_RIGHT:
		dy  = 1.0
	    elif event.key == K_UP:
		dx = -1.0
	    elif event.key == K_DOWN:
		dx = 1.0
	    elif event.key == K_d:
		move_x = 0.5
	    elif event.key == K_a:
		move_x = -0.5

	elif event.type == KEYUP:
	    if event.key == K_LEFT or event.key == K_RIGHT:
		dy = 0
	    elif event.key == K_UP or event.key == K_DOWN:
		dx = 0

	delta = (now - last_tick) / 1000.0
	last_tick = now

	stix.move(faces, delta)

	if dx != 0:
	    transform = rotate_x(transform, dx * delta)
	if dy != 0:
	    transform = rotate_y(transform, dy * delta)

        render.draw(mat_from_numpy(transform), stix.pos)
        pygame.display.flip()
        frames = frames+1


if __name__ == '__main__': main()
