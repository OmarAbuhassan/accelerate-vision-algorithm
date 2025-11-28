# accelerate-vision-algorithm


Project Refrence Document:

https://docs.google.com/document/d/1tCrjfi1lSoUkJQI9wUpu01KDAPt180MC/edit?pli=1


first build the image:

docker build -t openacc-env .

then run it:

docker run --gpus all -it --rm -v $(pwd):/app openacc-env


