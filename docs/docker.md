## Docker setup
This is an instruction on how to set up your docker environment with drake+uv.
Official development of drake uses bazel, which is suitable for large scale project (https://bazel.build/). 
However, if you are going to do projects in smaller scale, you might find it cumbersome to get used to bazel system (although it would be a good experience!).

uv (https://docs.astral.sh/uv/) is a super-fast python dependency resolver. It is 10x~100x faster than pip.

Both drake and uv offers official docker images, so for this project, we will combine these two images and setup our original container using docker compose.

Follow the instruction below.

### Setup
1. Clone the git repository.
   ```
    git clone git@github.com:Kota-Uchida/Drake-Final-Project.git
   ```
   If you have trouble creating ssh connection, refer to https://docs.github.com/en/authentication/connecting-to-github-with-ssh.

2. Build docker container. (Only in the first time)
    ```
    cd drake
    docker compose build
    ```
    If the `docker compose` command doesn't work, you should try `docker-compose` (with hyphen).

3. Run the docker container.
   ```
   docker compose up -d
   docker exec -it drake bash
   ```
   Or you can use the bash script
   ```
   bash docker-run.sh
   ```

4. Run `build.sh` to install the custom libraries.
   ```
   cd root/workspace/
   bash build.sh
   ```

Congratulations! Now you have entered the docker environment.

### Docker in VS Code
Here are some tips to use docker on VS Code.
1. Install extensions from "Container Tools", "Dev Containers", and "Docker".
2. Open the terminal with `Ctrl+@` and move to the project directory. 
3. Run `docker-compose up -d`.
4. Click the button in the left-bottom `><` and click `Attach to the running container`
5. VS Code will open a new tab for the container.