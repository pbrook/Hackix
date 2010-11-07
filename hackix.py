#!/usr/bin/env python
# Written by Paul Brook and ???
# Released under the GNU GPL v3.

from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import math


def resize((width, height)):
    if height==0:
        height=1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
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

class Point(Vector):
    def __init__(self, x, y, z):
	super(self.__class__, self).__init__(x, y, z)

# From point A to point B
class Edge(object):
    def __init__(self, a, b):
	self.a = a
	self.b = b
	self.split_edge = None

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
    def get_points(self):
	pa = self.a.a if self.own_a else self.a.b
	pb = self.b.a if self.own_b else self.b.b
	pc = self.c.a if self.own_c else self.c.b
	return (pa, pb, pc)

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

    def draw(self, transform):
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
	glLoadIdentity()
	glTranslatef(0.0, 0.0, -5.0)
	glMultMatrixf(transform)
	glVertexPointerf(self.vertexbuffer)
	glColorPointerf(self.colorbuffer)
	glDrawArrays(GL_TRIANGLES, 0, self.vertexcount)

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
    prev = edges.pop()
    # All edges are backwards
    origin = prev.b
    cur_point = prev.a
    while len(edges) > 1:
	for i in range(0, len(edges)):
	    if edges[i].b == cur_point:
		break;
	cur_point = edges[i].a
	new_edge = Edge(cur_point, origin)
	# FIXME do a new color
	yield Face(new_edge, prev, edges[i], (1.0, 1.0, 1.0), True, False, False)
	prev = new_edge
	del edges[i]

def do_split(faces, plane):
    class SplitEdge:
	def __init__(self, point, edge):
	    self.point = point
	    self.edge = edge
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
	    eb = f.a
	    opposite_edge = f.c
	    drop_point = bb
	elif bb == bc: # Point A
	    eb = f.a
	    ec = f.c
	    opposite_edge = f.b
	    drop_point = ba
	else:
	    raise Exception
	new_split_b = ec.split_edge is None
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
	    nf = Face(eb.split_edge.edge, opposite_edge, mid_edge, f.color, new_split_b, p[1][1], False)
	    newfaces.append(nf)
	    new_edge = Edge(ec.split_edge.point, eb.split_edge.point)
	    nf = Face(mid_edge, ec.split_edge.edge, new_edge, f.color, True, new_split_c, True);
	    newfaces.append(nf)
	else:
	    new_edge = Edge(eb.split_edge.point, ec.split_edge.point)
	    nf = Face(eb.split_edge.edge, new_edge, ec.split_edge.edge, f.color, new_split_b, True, new_split_c)
	    newfaces.append(nf)
	new_poly.append(new_edge)

    newfaces += list(make_poly(new_poly))
    return newfaces

def mul_mat3(matrix, matrix3):
    out = [0]*16
    out[0] = matrix3[0] * matrix[0] + matrix3[3] * matrix[1] + matrix3[6] * matrix[2]
    out[1] = matrix3[1] * matrix[0] + matrix3[4] * matrix[1] + matrix3[7] * matrix[2]
    out[2] = matrix3[2] * matrix[0] + matrix3[5] * matrix[1] + matrix3[8] * matrix[2]
    out[3] = matrix[3]
    out[4] = matrix3[0] * matrix[4] + matrix3[3] * matrix[5] + matrix3[6] * matrix[6]
    out[5] = matrix3[1] * matrix[4] + matrix3[4] * matrix[5] + matrix3[7] * matrix[6]
    out[6] = matrix3[2] * matrix[4] + matrix3[5] * matrix[5] + matrix3[8] * matrix[6]
    out[7] = matrix[7]
    out[8] = matrix3[0] * matrix[8] + matrix3[3] * matrix[9] + matrix3[6] * matrix[10]
    out[9] = matrix3[1] * matrix[8] + matrix3[4] * matrix[9] + matrix3[7] * matrix[10]
    out[10] = matrix3[2] * matrix[8] + matrix3[5] * matrix[9] + matrix3[8] * matrix[10]
    out[11] = matrix[11]
    out[12] = matrix[12]
    out[13] = matrix[13]
    out[14] = matrix[14]
    out[15] = matrix[15]
    return out

def rotate_x(matrix, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return mul_mat3(matrix, [1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c])

def rotate_y(matrix, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return mul_mat3(matrix, [c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c])

def rotate_z(matrix, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return mul_mat3(matrix, [c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0])


def main():

    video_flags = OPENGL|DOUBLEBUF
    
    pygame.init()
    pygame.display.set_mode((640,480), video_flags)

    resize((640,480))
    init()

    render = HackRender();
    frames = 0
    faces = create_cube();
    render.update_faces(faces)

    transform = [1.0, 0.0, 0.0, 0.0, \
		 0.0, 1.0, 0.0, 0.0, \
		 0.0, 0.0, 1.0, 0.0, \
		 0.0, 0.0, 0.0, 1.0];

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
		faces = do_split(faces, Plane(Vector(0.7,0.7,0.7), Vector(-1,-1,-1)))
		render.update_faces(faces)
	    elif event.key == K_LEFT:
		dy = -1.0
	    elif event.key == K_RIGHT:
		dy  = 1.0
	    elif event.key == K_UP:
		dx = -1.0
	    elif event.key == K_DOWN:
		dx = 1.0
	elif event.type == KEYUP:
	    if event.key == K_LEFT or event.key == K_RIGHT:
		dy = 0
	    elif event.key == K_UP or event.key == K_DOWN:
		dx = 0

	delta = (now - last_tick) / 1000.0
	last_tick = now
	if dx != 0:
	    transform = rotate_x(transform, dx * delta)
	if dy != 0:
	    transform = rotate_y(transform, dy * delta)

        render.draw(transform)
        pygame.display.flip()
        frames = frames+1


if __name__ == '__main__': main()
