require('premake5-cuda')

workspace "LinearSolver"
    architecture "x64"

    configurations
    {
        "Debug",
        "Release"
    }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "LinearSolver"
    location "LinearSolver"     -- 定义项目生成位置
    kind "ConsoleApp"
    language "C++"

    targetdir("bin/" .. outputdir .. "/%{prj.name}")        -- 定义二进制文件生成位置
    objdir("bin-int/" .. outputdir .. "/%{prj.name}")       -- 定义obj文件生成位置

    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp"
    }

    buildcustomizations "BuildCustomizations/CUDA 11.4"

    cudaFiles 
    {
        "%{prj.name}/src/*.cu"
    }
    
    cudaMaxRegCount "32"       -- 定义最大寄存器数量

    cudaCompilerOptions {"-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
                     "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
                     "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
                     "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86", "-t0"}    

    -- 在windows，它会被自动连接，在linux他必须手动链接
    if os.target() == "linux" then 
        linkoptions {"-L/usr/local/cuda/lib64 -lcudart"}
    end

    includedirs
    {
        "%{prj.name}/src",
        "$(CUDA_PATH)/include"
    }

    libdirs
    {
        "$(CUDA_PATH)/lib/x64"
    }

    links
    {
        "cuda"
    }

    filter "system:windows"
        staticruntime "On"
        systemversion "latest"

        defines 
        { 
            "_CONSOLE"
        }
    
    filter "configurations:Debug"
        symbols "On"

    filter "configurations:Release"
        optimize "On"
        cudaFastMath "On"
		
        