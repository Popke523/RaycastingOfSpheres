#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "kernel.h"

#include <iostream>

#include "random_objects_generator.h"

#define CHECK_CUDA_ERR(x)                                         \
    do {                                                          \
        cudaError_t err = x;                                      \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr,                                       \
                    "CUDA Error at %s:%d -> %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// Global Variables
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

unsigned int texture;
cudaGraphicsResource *m_TextureResource;
cudaArray *array;
cudaResourceDesc desc;
cudaSurfaceObject_t surface;

camera cam;

// Current Width and Height
int currentWidth = SCR_WIDTH;
int currentHeight = SCR_HEIGHT;

const int N_SPHERES = 100;
const int N_LIGHTS = 10;

const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;

out vec3 ourColor;
out vec2 TexCoord;

void main()
{
	gl_Position = vec4(aPos, 1.0);
	ourColor = aColor;
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}
)";

const char *fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture sampler
uniform sampler2D texture1;

void main()
{
	FragColor = texture(texture1, TexCoord);
}
)";

// Function to compile shaders
GLuint compileShader(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
    }
    return shader;
}

int main()
{
    cudaSetDevice(0);

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // Set up ImGui backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Set up ImGui style
    ImGui::StyleColorsDark();

    // build and compile our shader zprogram
    // ------------------------------------
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    // Allocate texture storage
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, currentWidth, currentHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Register the texture with CUDA
    CHECK_CUDA_ERR(cudaGraphicsGLRegisterImage(&m_TextureResource, texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore));

    // Map the CUDA resource
    CHECK_CUDA_ERR(cudaGraphicsMapResources(1, &m_TextureResource, 0));

    CHECK_CUDA_ERR(cudaGraphicsSubResourceGetMappedArray(&array, m_TextureResource, 0, 0));

    // Setup CUDA resource descriptor
    memset(&desc, 0, sizeof(cudaResourceDesc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;

    // Create a CUDA surface object
    CHECK_CUDA_ERR(cudaCreateSurfaceObject(&surface, &desc));
   
    srand(time(nullptr));

    cam.position = make_float3(0.0f, 0.0f, 10.0f);
    cam.fov_degrees = 90.0f;
    cam.pitch_degrees = 0.0f;
    cam.yaw_degrees = 0.0f;

	float brightness = 1.0f;
    float kd = 1.0f;
    float ks = 1.0f;

    sphere spheres[N_SPHERES];
    lightSource lightSources[N_LIGHTS];
    for (int i = 0; i < N_SPHERES; i++)
    {
        spheres[i] = random_sphere(-5.0f, 5.0f, 0.1f, 0.4f);	
    }
    for (int i = 0; i < N_LIGHTS; i++)
    {
        lightSources[i] = random_light_source(-10.0f, 10.0f);
    }

	sphere *deviceSpheres;
	cudaMalloc(&deviceSpheres, sizeof(spheres));
	cudaMemcpy(deviceSpheres, &spheres, sizeof(spheres), cudaMemcpyHostToDevice);

    lightSource *deviceLightSources;
    cudaMalloc(&deviceLightSources, sizeof(lightSources));
	cudaMemcpy(deviceLightSources, &lightSources, sizeof(lightSources), cudaMemcpyHostToDevice);

    glm::mat4 lightRotations[N_LIGHTS];
	for (int i = 0; i < N_LIGHTS; i++)
	{
		float yaw = random_float(-0.5f, 0.5f);
		float pitch = random_float(-0.5f, 0.5f);
		float roll = random_float(-0.5f, 0.5f);

        glm::mat4 rotationMatrix = glm::mat4(1.0f);
		rotationMatrix = glm::rotate(rotationMatrix, glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));
		rotationMatrix = glm::rotate(rotationMatrix, glm::radians(pitch), glm::vec3(1.0f, 0.0f, 0.0f));
		rotationMatrix = glm::rotate(rotationMatrix, glm::radians(roll), glm::vec3(0.0f, 0.0f, 1.0f));

        lightRotations[i] = rotationMatrix;
	}


    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        renderTestKernelLauncher(surface, currentWidth, currentHeight, cam, deviceSpheres, N_SPHERES, deviceLightSources, N_LIGHTS, brightness, kd, ks);
        //cudaGraphicsUnmapResources(1, &m_TextureResource);

        // render
        // ------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, texture);

        glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


        // Start new ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create GUI
        ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_Once);
		ImGui::SetNextWindowSize(ImVec2(320.0f, 240.0f), ImGuiCond_Once);
        ImGui::Begin("Simple GUI");

        ImGui::Text("Frame rate: %.1f FPS", ImGui::GetIO().Framerate);

		// imgui sliders for camera settings
		ImGui::SliderFloat("Camera X", &cam.position.x, -10.0f, 10.0f);
		ImGui::SliderFloat("Camera Y", &cam.position.y, -10.0f, 10.0f);
		ImGui::SliderFloat("Camera Z", &cam.position.z, -10.0f, 10.0f);
		ImGui::SliderFloat("Camera Pitch", &cam.pitch_degrees, -90.0f, 90.0f);
		ImGui::SliderFloat("Camera Yaw", &cam.yaw_degrees, -180.0f, 180.0f);
        ImGui::SliderFloat("Camera FOV", &cam.fov_degrees, 30.0f, 150.0f);
        ImGui::SliderFloat("Brightness", &brightness, 0.0f, 1.0f);
        ImGui::SliderFloat("kd", &kd, 0.0f, 1.0f);
        ImGui::SliderFloat("ks", &ks, 0.0f, 1.0f);

        ImGui::SetCursorPosX(0.0f);
        ImGui::SetCursorPosY(0.0f);

        ImGui::End();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


        // bind Texture
        // glBindTexture(GL_TEXTURE_2D, texture);

        // render container
        //glBindVertexArray(VAO);
        //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        for(int i = 0; i < N_LIGHTS; i++)
		{
			glm::vec3 position = glm::vec3(lightSources[i].position.x, lightSources[i].position.y, lightSources[i].position.z);
            position = glm::vec3(lightRotations[i] * glm::vec4(position, 1.0f));
            lightSources[i].position = make_float3(position.x, position.y, position.z);
		}

		cudaMemcpy(deviceLightSources, lightSources, sizeof(lightSources), cudaMemcpyHostToDevice);

    }

    CHECK_CUDA_ERR(cudaGraphicsUnmapResources(1, &m_TextureResource, 0));
    CHECK_CUDA_ERR(cudaGraphicsUnregisterResource(m_TextureResource));

	// Cleanup
	CHECK_CUDA_ERR(cudaFree(deviceSpheres));
    CHECK_CUDA_ERR(cudaFree(deviceLightSources));

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float yaw = glm::radians(cam.yaw_degrees);
    float pitch = glm::radians(cam.pitch_degrees);
    float3 front;
	front.x = -sin(yaw) * cos(pitch);
	front.y = sin(pitch);
	front.z = -cos(yaw) * cos(pitch);

    float cameraSpeed = 0.1f;
    float3 right = normalize(cross(front, make_float3(0.0f, 1.0f, 0.0f)));


    // Arrow keys for rotation
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        cam.pitch_degrees -= cameraSpeed * 10.0f; // Adjust rotation speed
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        cam.pitch_degrees += cameraSpeed * 10.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        cam.yaw_degrees += cameraSpeed * 10.0f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        cam.yaw_degrees -= cameraSpeed * 10.0f;

    // WASD for movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cam.position += cameraSpeed * front;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cam.position -= cameraSpeed * front;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cam.position -= right * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cam.position += right * cameraSpeed;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // Update the OpenGL viewport
    glViewport(0, 0, width, height);

    // If the window size hasn't changed, no need to update
    if (width == currentWidth && height == currentHeight)
        return;

    // Update current width and height
    currentWidth = width;
    currentHeight = height;

    // Unmap and unregister the existing CUDA resource
    CHECK_CUDA_ERR(cudaGraphicsUnmapResources(1, &m_TextureResource, 0));
    CHECK_CUDA_ERR(cudaGraphicsUnregisterResource(m_TextureResource));

    // Delete the existing texture
    glDeleteTextures(1, &texture);

    // Create a new texture with updated size
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // Set texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    // Allocate texture storage with the new dimensions
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Register the new texture with CUDA
    CHECK_CUDA_ERR(cudaGraphicsGLRegisterImage(&m_TextureResource, texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore));

    // Map the CUDA resource
    CHECK_CUDA_ERR(cudaGraphicsMapResources(1, &m_TextureResource, 0));


    // Get the CUDA array from the graphics resource
    CHECK_CUDA_ERR(cudaGraphicsSubResourceGetMappedArray(&array, m_TextureResource, 0, 0));

    // Setup CUDA resource descriptor
    memset(&desc, 0, sizeof(cudaResourceDesc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;

    // Create a new CUDA surface object
    CHECK_CUDA_ERR(cudaCreateSurfaceObject(&surface, &desc));

    std::cout << "Framebuffer resized to " << width << "x" << height << std::endl;
}
