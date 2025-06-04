#include <gtk/gtk.h>
#include <string.h>

typedef struct {
    GtkWidget *install_entry;
    GtkWidget *remove_entry;
    GtkWidget *output_view;
    char *model_path;
} AppWidgets;

static void append_text(GtkTextView *view, const char *text) {
    GtkTextBuffer *buffer = gtk_text_view_get_buffer(view);
    GtkTextIter end;
    gtk_text_buffer_get_end_iter(buffer, &end);
    gtk_text_buffer_insert(buffer, &end, text, -1);
}

static void run_command(const char *cmd, GtkTextView *output) {
    gchar *stdout_buf = NULL;
    gchar *stderr_buf = NULL;
    gint exit_status = 0;

    append_text(output, "\n> ");
    append_text(output, cmd);
    append_text(output, "\n");

    if (g_spawn_command_line_sync(cmd, &stdout_buf, &stderr_buf, &exit_status, NULL)) {
        if (stdout_buf) {
            append_text(output, stdout_buf);
            g_free(stdout_buf);
        }
        if (stderr_buf) {
            append_text(output, stderr_buf);
            g_free(stderr_buf);
        }
    } else {
        append_text(output, "Failed to run command\n");
    }
}

static void apply_actions(GtkButton *button, gpointer data) {
    AppWidgets *widgets = (AppWidgets *)data;
    const char *install = gtk_entry_get_text(GTK_ENTRY(widgets->install_entry));
    const char *remove = gtk_entry_get_text(GTK_ENTRY(widgets->remove_entry));

    if (strlen(install) > 0) {
        char *cmd = g_strdup_printf("apt-get update && apt-get install -y %s", install);
        run_command(cmd, GTK_TEXT_VIEW(widgets->output_view));
        g_free(cmd);
    }

    if (strlen(remove) > 0) {
        char *cmd = g_strdup_printf("apt-get remove -y %s", remove);
        run_command(cmd, GTK_TEXT_VIEW(widgets->output_view));
        g_free(cmd);
    }
}

static void file_selected(GtkFileChooserButton *button, gpointer data) {
    AppWidgets *widgets = (AppWidgets *)data;
    char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(button));
    if (filename) {
        if (widgets->model_path) {
            g_free(widgets->model_path);
        }
        widgets->model_path = filename; /* keep path for later use */
        g_print("Selected model: %s\n", filename);
    }
}

static void run_model(GtkButton *button, gpointer data) {
    AppWidgets *widgets = (AppWidgets *)data;
    if (widgets->model_path) {
        run_command(widgets->model_path, GTK_TEXT_VIEW(widgets->output_view));
    } else {
        append_text(GTK_TEXT_VIEW(widgets->output_view), "No model selected\n");
    }
}

static void activate(GtkApplication* app, gpointer user_data) {
    GtkWidget *window;
    GtkWidget *vbox;
    GtkWidget *file_button;
    GtkWidget *run_model_btn;
    GtkWidget *install_entry;
    GtkWidget *remove_entry;
    GtkWidget *output_view;
    GtkWidget *apply_btn;
    AppWidgets *widgets = g_new0(AppWidgets, 1);

    window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "Docker AI");
    gtk_window_set_default_size(GTK_WINDOW(window), 600, 400);

    vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    file_button = gtk_file_chooser_button_new("Select AI model", GTK_FILE_CHOOSER_ACTION_OPEN);
    g_signal_connect(file_button, "file-set", G_CALLBACK(file_selected), widgets);
    gtk_box_pack_start(GTK_BOX(vbox), file_button, FALSE, FALSE, 0);

    run_model_btn = gtk_button_new_with_label("Run Model");
    gtk_box_pack_start(GTK_BOX(vbox), run_model_btn, FALSE, FALSE, 0);
    g_signal_connect(run_model_btn, "clicked", G_CALLBACK(run_model), widgets);

    install_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(install_entry), "Packages to install/update (space separated)");
    gtk_box_pack_start(GTK_BOX(vbox), install_entry, FALSE, FALSE, 0);

    remove_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(remove_entry), "Packages to remove (space separated)");
    gtk_box_pack_start(GTK_BOX(vbox), remove_entry, FALSE, FALSE, 0);

    apply_btn = gtk_button_new_with_label("Apply");
    gtk_box_pack_start(GTK_BOX(vbox), apply_btn, FALSE, FALSE, 0);

    output_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(output_view), FALSE);
    gtk_box_pack_start(GTK_BOX(vbox), output_view, TRUE, TRUE, 0);

    widgets->install_entry = install_entry;
    widgets->remove_entry = remove_entry;
    widgets->output_view = output_view;
    widgets->model_path = NULL;

    g_signal_connect(apply_btn, "clicked", G_CALLBACK(apply_actions), widgets);

    gtk_widget_show_all(window);
}

int main(int argc, char **argv) {
    GtkApplication *app;
    int status;

    app = gtk_application_new("com.example.dockerai", G_APPLICATION_FLAGS_NONE);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return status;
}
