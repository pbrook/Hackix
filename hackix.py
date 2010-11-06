#!/usr/bin/env python
# Written by Paul Brook and ???
# Released under the GNU GPL v3.

from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *


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

class Plane(object):
    def __init__(self, p, n):
	self.p = p
	self.n = n
    def behind(self, v):
	return self.n.dot_product(v - self.p) < 0
    def intersect(self, p, n):
	p0 = self.p - p
	v = self.n.dot_product(p0) / self.n.dot_product(n)
	return Point(p.x + v * n.x, p.y + v * n.y, p.z + v * n.z)

class HackRender(object):
    def update_faces(self, faces):
	self.vertexbuffer=[]
	self.colorbuffer=[]
	self.vertexcount = 0
	for f in faces:
	    print self.vertexcount / 3
	    for (e, own) in ((f.a, f.own_a), (f.b, f.own_b), (f.c, f.own_c)):
		p = e.a if own else e.b
		self.vertexbuffer.append((p.x, p.y, p.z))
		self.colorbuffer.append(f.color)
	    self.vertexcount += 3

    def draw(self, angle1, angle2):
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
	glLoadIdentity()
	glTranslatef(0.0, 0.0, -10.0)
	glRotatef(angle2, 1.0, 0.0, 0.0)
	glRotatef(angle1, 0.0, 1.0, 0.0)
	glVertexPointerf(self.vertexbuffer)
	glColorPointerf(self.colorbuffer)
	glDrawArrays(GL_TRIANGLES, 0, self.vertexcount)

def create_cube():
    points=[]
    n = 0
    for x in (1.0, -1.0):
	for y in (1.0, -1.0):
	    for z in (1.0, -1.0):
		points.append(Point(x, y, z))
		print n, points[-1]
		n = n + 1
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
    first = edges.pop()
    origin = first[0]
    cur = first[1]
    print origin
    while len(edges) > 1:
	print cur
	print len(edges)
	for i in range(0, len(edges)):
	    print edges[i][0], edges[i][1]
	    if edges[i][0] == cur:
		break;
	print i
	if edges[i][1] == origin:
	    break;
	yield Face(origin, cur, edges[i][1], (1.0, 1.0, 1.0))
	cur == edges[i][1]
	del edges[i]
    
def do_split(faces, plane):
    newfaces = []
    edges = []
    for f in faces:
	ba = plane.behind(f.a)
	bb = plane.behind(f.b)
	bc = plane.behind(f.c)
	if ba == bb and bb == bc:
	    if not ba:
		newfaces.append(f)
	    continue
	if ba == bb:
	    test = bc
	    pa = f.c
	    pb = f.a
	    pc = f.b
	elif bb == bc:
	    test = ba
	    pa = f.a
	    pb = f.b
	    pc = f.c
	elif bc == ba:
	    test = bc
	    pa = f.b
	    pb = f.c
	    pc = f.a
	else:
	    raise Exception
	ib = plane.intersect(pa, pb - pa)
	ic = plane.intersect(pa, pc - pa)
	print 'Make Edge'
	if test:
	    # drop point
	    newfaces.append(Face(ib, pb, pc, f.color))
	    newfaces.append(Face(ib, pc, ic, f.color))
	    edges.append((ib, ic))
	else:
	    # keep point
	    newfaces.append(Face(pa, ib, ic, f.color))
	    edges.append((ic, ib))
    newfaces += list(make_poly(edges))
    return newfaces

def main():

    video_flags = OPENGL|DOUBLEBUF
    
    pygame.init()
    pygame.display.set_mode((640,480), video_flags)

    resize((640,480))
    init()

    render = HackRender();
    frames = 0
    ticks = pygame.time.get_ticks()
    faces = create_cube();
    render.update_faces(faces)

    while 1:
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break

	if event.type == KEYDOWN and event.key == K_SPACE:
	    faces = do_split(faces, Plane(Vector(0.7,0.7,0.7), Vector(-1,-1,-1)))
	    render.update_faces(faces)
        
        render.draw(frames, frames / 10.0)
        pygame.display.flip()
        frames = frames+1

    print "fps:  %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks))


if __name__ == '__main__': main()
