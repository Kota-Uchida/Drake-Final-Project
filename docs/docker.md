## Docker setup
This is an instruction on how to set up your docker environment with drake+uv.
Official development of drake uses bazel, which is suitable for large scale project (https://bazel.build/). 
However, if you are going to do projects in smaller scale, you might find it cumbersome to get used to bazel system (although it would be a good experience!).

uv (https://docs.astral.sh/uv/) is a super-fast python dependency resolver. It is 10x~100x faster than pip.

Both drake and uv offers official docker images, so for this project, we will combine these two images and setup our original container using docker compose.

Follow the instruction below.

1. 