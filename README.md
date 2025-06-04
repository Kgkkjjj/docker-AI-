# Docker AI

This repository contains a minimal example of a C-based GUI application that
runs inside Docker. The GUI is built using GTK 3 and displays three text areas
for experimentation. It does not implement real AI features but serves as a
starting point for integrating tools to install, update or manage packages
inside a container. Users can upload an AI model through the interface; when a
file is selected the application prints the path as a placeholder for running
the model inside Docker.

## Building and running

1. Build the Docker image:
   ```sh
   docker build -t docker-ai .
   ```
2. Run the container (mount a directory containing your AI model if needed):
   ```sh
   docker run --rm -it -v /path/to/models:/models docker-ai
   ```

This will launch the GTK interface within the container. Use the “Select AI
model” button to choose a model file from the mounted directory. The selected
path is printed to the container logs.
