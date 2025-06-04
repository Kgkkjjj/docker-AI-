# Docker AI

This project provides a C/GTK GUI that runs inside Docker. The interface lets you select an AI model file and manage packages inside the container.

The window contains:

- **AI model chooser** – pick a model file from a mounted directory.
- **Run Model** – executes the chosen model file and shows its output.
- **Install/Update field** – enter package names to install or upgrade.
- **Remove field** – enter package names to remove.
- **Output area** – shows output from the package commands and model.

When you press **Run Model** the selected file is executed as a script or binary. Press **Apply** to run `apt-get` with the package fields. All output is shown in the text area.

## Building and running

1. Build the Docker image:
   ```sh
   docker build -t docker-ai .
   ```
2. Run the container (mount a directory containing your AI model if needed):
   ```sh
   docker run --rm -it -v /path/to/models:/models docker-ai
   ```

This launches the GUI inside the container. Use the fields to manage packages as needed.
