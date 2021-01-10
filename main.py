import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from ctypes import *
import numpy as np
from math import *
from imgui import *

from numpy.core.numeric import cross, normalize_axis_index

vertexSource = """
#version 410 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexcoord;

out vec2 texcoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    texcoord = aTexcoord;
}
"""

fragmentSource = """
#version 410 core

in vec2 texcoord;

uniform vec3 camPos;
uniform vec3 camFront;
uniform vec3 camUp;

uniform float power;
uniform int iterations;

float getDist(vec3 p) {
    //vec4 s = vec4(0.0,0.5,0.0,1.0);
    //float sphereDist = length(p-s.xyz)-s.w;
    //float planeDist = p.y;
    //return min(sphereDist, planeDist);
    vec3 w = p;
    float m = dot(w,w);

    vec4 trap = vec4(abs(w),m);
    float dz = 1.0;

    for (int i = 0; i < iterations; i++) {
        dz = power*pow(sqrt(m),power-1.0)*dz+1.0;

        float r = length(w);
        float b = power*acos(w.y/r);
        float a = power*atan(w.x,w.z);
        w = p + pow(r,power) * vec3(sin(b)*sin(a), cos(b), sin(b)*cos(a));
        trap = min(trap, vec4(abs(w), m));

        m = dot(w,w);
        if (m > 4.0)
            break;
    }
    return 0.25*log(m)*sqrt(m)/dz;
}

float rayMarch(vec3 ro, vec3 rd) {
    float d0 = 0.0;
    for (int i = 0; i < 100; i++) {
        vec3 p = ro+rd*d0;
        float ds = getDist(p);
        d0 += ds;
        if (d0 > 100.0 || ds < 0.01) break;
    }
    return d0;
}

float rayMarchRef(vec3 ro, vec3 rd) {
    float d0 = 0.0;
    for (int i = 0; i < 75; i++) {
        vec3 p = ro+rd*d0;
        float ds = getDist(p);
        d0 += ds;
        if (d0 > 100.0 || abs(ds) < 0.02) break;
    }
    return d0;
}

vec3 getNormal(vec3 p) {
    float d = getDist(p);
    vec2 e = vec2(.01,0.0);
    vec3 n = d - vec3(
        getDist(p-e.xyy),
        getDist(p-e.yxy),
        getDist(p-e.yyx)
    );
    return normalize(n);
}

float getLight(vec3 p) {
    vec3 lightPos = vec3(0.0,10.0,1.0);
    vec3 l = normalize(lightPos-p);
    vec3 n = getNormal(p);

    float dif = dot(n,l);
    dif = clamp(dif,0.0,1.0);

    float d = rayMarch(p+n*0.01*2.0,l);
    if (d < length(lightPos-p)) dif *= .1;

    return dif;
}

vec3 getCameraRayDir(vec2 coord) {
    vec3 vdir = normalize(coord.x*-normalize(cross(vec3(0.0,1.0,0.0), camFront))+coord.y*camUp+camFront*2.0);
    return vdir;
}

float getCoC(float depth, float focalPlane) {
    float focalLength = 0.02;
    float aperture = min(1.0, focalPlane * focalPlane);
    return abs(aperture * (focalLength * (focalPlane - depth)) / (depth * (focalPlane - focalLength)));
}

vec3 mapToColor(float t) {
    float r = 9.0*(1.0-t)*t*t*t;
    float g = 15.0*(1.0-t)*(1.0-t)*t*t;
    float b = 8.5*(1.0-t)*(1.0-t)*(1.0-t)*t;
    return vec3(r, g, b);
}

float getShadow(vec3 ro, vec3 lp, float k) {
    const int maxIterShad = 24;

    vec3 rd = lp - ro;

    float shade = 1.0;
    float dist = 0.002;
    float end = max(length(rd), 0.01);
    float stepDist = end/float(maxIterShad);
    rd /= end;

    for (int i = 0; i < maxIterShad; i++) {
        float h = getDist(ro+rd*dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist));
        dist += clamp(h, 0.02, 0.25);
        if (h < 0.0 || dist > end) break;
    }
    return min(max(shade, 0.0) + 0.25, 1.0);
}

// Some Rayleigh scattering stuff

int iSteps = 8;
int jSteps = 8;

vec2 rsi(vec3 ro, vec3 rd, float sr) {
    float a = dot(rd, rd);
    float b = 2.0*dot(rd,ro);
    float c = dot(ro,ro)-(sr*sr);
    float d = (b*b)-4.0*a*c;
    if (d < 0.0) return vec2(1e5,-1e5);
    return vec2(
        (-b-sqrt(d))/(2.0*a),
        (-b+sqrt(d))/(2.0*a)
    );
}

vec3 atmosphere(vec3 r, vec3 ro, vec3 pSun, float iSun, float rPlanet, float rAtmos, vec3 kRlh, float kMie, float shRlh, float shMie, float g) {
    pSun = normalize(pSun);
    r = normalize(r);

    vec2 p = rsi(ro, r, rAtmos);
    if (p.x > p.y) return vec3(0.0,0.0,0.0);
    p.y = min(p.y, rsi(ro, r, rPlanet).x);
    float iStepSize = (p.y-p.x)/float(iSteps);

    float iTime = 0.0;

    vec3 totalRlh = vec3(0.0,0.0,0.0);
    vec3 totalMie = vec3(0.0,0.0,0.0);

    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    float mu = dot(r, pSun);
    float mumu = mu*mu;
    float gg = g*g;
    float pRlh = 3.0/(16.0*3.14159265359)*(1.0+mumu);
    float pMie = 3.0/(8.0*3.14159265359)*((1.0-gg)*(mumu+1.0))/(pow(1.0+gg-2.0*mu*g,1.5)*(2.0+gg));

    for (int i = 0; i < iSteps; i++) {
        vec3 iPos = ro+r*(iTime+iStepSize*0.5);
        
        float iHeight = length(iPos)-rPlanet;

        float odStepRlh = exp(-iHeight/shRlh)*iStepSize;
        float odStepMie = exp(-iHeight/shMie)*iStepSize;

        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        float jStepSize = rsi(iPos, pSun, rAtmos).y / float(jSteps);
        
        float jTime = 0.0;

        float jOdRlh = 0.0;
        float jOdMie = 0.0;

        for (int j = 0; j < jSteps; j++) {
            vec3 jPos = iPos+pSun*(jTime+jStepSize*0.5);

            float jHeight = length(jPos)-rPlanet;

            jOdRlh += exp(-jHeight/shRlh)*jStepSize;
            jOdMie += exp(-jHeight/shMie)*jStepSize;

            jTime += jStepSize;
        }

        vec3 attn = exp(-(kMie*(iOdMie+jOdMie)+kRlh*(iOdRlh+jOdRlh)));

        totalRlh += odStepRlh * attn;
        totalMie += odStepMie * attn;

        iTime += iStepSize;
    }

    return iSun*(pRlh*kRlh*totalRlh+pMie*kMie*totalMie);
}

void main() {
    vec2 coord = (gl_FragCoord.xy/vec2(720.0,720.0))*2.0-1.0;
    vec3 ro = camPos;
    vec3 rd = getCameraRayDir(coord);
    float d = rayMarch(ro, rd);
    vec3 p = ro+rd*d;
    vec3 col = vec3(getLight(p));
    vec3 skyCol;
    if (col == vec3(0.0,0.0,0.0) && getDist(p) > 100.0) {
        col = atmosphere(rd, vec3(0.0,6372e3,0.0),vec3(0.0,0.1,-1.0),22.0,6371e3,6471e3,vec3(5.5e-6,13.0e-6, 22.4e-6), 21e-6, 8e3, 1.2e3, 0.758);
        col = 1.0-exp(-1.0*col);
    }
    //col *= vec3(getShadow(ro+getNormal(p)*0.0015,vec3(0.0,10.0,0.0), 16.0));
    gl_FragColor = vec4(col, 1.0);
}
"""

camPos = [0.0,0.0,2.0]

camFront = [0,0,-1]
camUp = [0,1,0]

power = 8.0

iterations = 4

cameraSpeed = 0.05

def processInput(window):
    global camPos, power, iterations, cameraSpeed
    deltaTime = 0.0
    lastFrame = 0.0
    currentFrame = glfw.get_time()
    deltaTime = currentFrame - lastFrame
    lastFrame = currentFrame
    if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS):
        camPos[0] += cameraSpeed*camFront[0]
        camPos[1] += cameraSpeed*camFront[1]
        camPos[2] += cameraSpeed*camFront[2]
    if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS):
        camPos[0] -= cameraSpeed*camFront[0]
        camPos[1] -= cameraSpeed*camFront[1]
        camPos[2] -= cameraSpeed*camFront[2]
    if (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS):
        camPos -= cross(camFront, camUp)*cameraSpeed
    if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS):
        camPos += cross(camFront, camUp)*cameraSpeed
    if (glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS):
        camPos[1] += cameraSpeed
    if (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS):
        camPos[1] -= cameraSpeed

    if (glfw.get_key(window, glfw.KEY_R) == glfw.PRESS):
        camPos = [0.0,0.0,2.0]

    if (glfw.get_key(window, glfw.KEY_O) == glfw.PRESS):
        cameraSpeed /= 2.0
    if (glfw.get_key(window, glfw.KEY_I) == glfw.PRESS):
        cameraSpeed *= 2.0

    if (glfw.get_key(window, glfw.KEY_KP_ADD) == glfw.PRESS):
        power += 0.1
        print(power)
    if (glfw.get_key(window, glfw.KEY_KP_SUBTRACT) == glfw.PRESS):
        power -= 0.1
        print(power)

    if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
        iterations += 1
    if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
        iterations -= 1

    if (glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
        glfw.set_window_should_close(window, True)

lastX = 720/2
lastY = 720/2

yaw = -90.0
pitch = 0.0

firstMouse = True
isDragging = False

def mouseButtonCallback(window, button, action, mods):
    global isDragging
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        isDragging = True
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        isDragging = False

def mouseCallback(window, xpos, ypos):
    global lastX, lastY, yaw, pitch, camFront, firstMouse
    if (firstMouse):
        lastX = xpos
        lastY = ypos
        firstMouse = False
    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.1
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset
    
    direction = [0,0,0]
    direction[0] = cos(radians(yaw)) * cos(radians(pitch))
    direction[1] = sin(radians(pitch))
    direction[2] = sin(radians(yaw)) * cos(radians(pitch))
    camFront = direction

def main():
    global camPos
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(720, 720, "UniFract 3D", None, None)
    if not window:
        glfw.terminate()

    # Set Callbacks

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_cursor_pos_callback(window, mouseCallback)
    #glfw.set_mouse_button_callback(window, mouseButtonCallback)

    vertices = np.array([
        -1.0,  1.0,   0.0,  1.0,
        -1.0, -1.0,   0.0,  0.0,
         1.0, -1.0,   1.0,  0.0,

        -1.0,  1.0,   0.0,  1.0,
         1.0, -1.0,   1.0,  0.0,
         1.0,  1.0,   1.0,  1.0
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, c_void_p(0))
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, c_void_p(8))
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    shaderProgram = compileProgram(compileShader(vertexSource, GL_VERTEX_SHADER), compileShader(fragmentSource, GL_FRAGMENT_SHADER))

    glUseProgram(shaderProgram)

    glUniform3f(glGetUniformLocation(shaderProgram, "camPos"), camPos[0], camPos[1], camPos[2])
    glUniform3f(glGetUniformLocation(shaderProgram, "camFront"), camFront[0], camFront[1], camFront[2])
    glUniform3f(glGetUniformLocation(shaderProgram, "camUp"), camUp[0], camUp[1], camUp[2])
    glUniform1f(glGetUniformLocation(shaderProgram, "power"), power)
    glUniform1i(glGetUniformLocation(shaderProgram, "iterations"), iterations)

    while not glfw.window_should_close(window):
        processInput(window)

        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shaderProgram)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        glUniform3f(glGetUniformLocation(shaderProgram, "camPos"), camPos[0], camPos[1], camPos[2])
        glUniform3f(glGetUniformLocation(shaderProgram, "camFront"), camFront[0], camFront[1], camFront[2])
        glUniform3f(glGetUniformLocation(shaderProgram, "camUp"), camUp[0], camUp[1], camUp[2])
        glUniform1f(glGetUniformLocation(shaderProgram, "power"), power)
        glUniform1i(glGetUniformLocation(shaderProgram, "iterations"), iterations)

        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()

print("""
W A S D | Move around
R | Reset camera position
Up arrow | Increase iteration count
Down arrow | Decrease iteration count
+ | Increase power
- | Decrease power
I | Increase camera speed
O | Decrease camera speed

Esc | Close UniFract 3D
""")

main()
