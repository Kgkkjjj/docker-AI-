# Docker AI Tool

This project provides a simple web interface for managing packages and running commands inside a Docker container. The UI is served by a small Flask application and lives in the `web/` folder.

The web page offers several actions:

- **Install packages** – runs `apt-get install` inside the container.
- **Remove packages** – runs `apt-get remove` inside the container.
- **Run command** – executes an arbitrary shell command and shows the output.
- **Chat with AI** – send prompts to an OpenAI model (requires `OPENAI_API_KEY`).

All output from these operations is displayed in the browser.

## Building and Running

1. Build the Docker image:
   ```sh
   docker build -t docker-ai .
   ```
2. Run the container and expose the web interface:
   ```sh
   docker run --rm -p 5000:5000 docker-ai
   ```
3. Open `http://localhost:5000` in your browser to access the tool.

To enable the chat feature, supply an API key when starting the container:

```sh
docker run --rm -p 5000:5000 -e OPENAI_API_KEY=sk-... docker-ai
```

The container needs network access to install packages. Feel free to modify the Dockerfile to include additional dependencies or mount volumes if required.
