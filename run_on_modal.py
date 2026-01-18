import modal
import subprocess

# 定义云端环境：安装 gcc
image = modal.Image.debian_slim().apt_install("gcc")

app = modal.App("matrix-lab")

# 将本地 C 文件挂载到云端并执行
@app.function(image=image, cpu=16.0)
def run_bench():
    # 编译
    subprocess.run("gcc -O3 matrix_lab.c -o matrix_lab -lpthread -lm", shell=True, check=True)
    # 运行
    result = subprocess.run("./matrix_lab", shell=True, capture_output=True, text=True)
    print(result.stdout)

@app.local_entrypoint()
def main():
    run_bench.remote()
