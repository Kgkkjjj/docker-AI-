#include <gtk/gtk.h>

/* Called when the user selects an AI model file. The current
 * implementation only prints the path but this is where integration
 * with the model would occur. */
static void file_selected(GtkFileChooserButton *button, gpointer user_data) {
    char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(button));
    if (filename) {
        g_print("Selected model: %s\n", filename);
        /* TODO: invoke the model inside Docker */
        g_free(filename);
    }
}

static void activate (GtkApplication* app, gpointer user_data) {
    GtkWidget *window;
    GtkWidget *vbox;
    GtkWidget *file_button;
    GtkWidget *box1;
    GtkWidget *box2;
    GtkWidget *box3;

    window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "Docker AI");
    gtk_window_set_default_size(GTK_WINDOW(window), 600, 400);

    vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    file_button = gtk_file_chooser_button_new("Select AI model", GTK_FILE_CHOOSER_ACTION_OPEN);
    g_signal_connect(file_button, "file-set", G_CALLBACK(file_selected), NULL);
    gtk_box_pack_start(GTK_BOX(vbox), file_button, FALSE, FALSE, 0);

    box1 = gtk_text_view_new();
    box2 = gtk_text_view_new();
    box3 = gtk_text_view_new();

    gtk_box_pack_start(GTK_BOX(vbox), box1, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vbox), box2, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vbox), box3, TRUE, TRUE, 0);

    gtk_widget_show_all(window);
}

int main (int argc, char **argv) {
    GtkApplication *app;
    int status;

    app = gtk_application_new("com.example.dockerai", G_APPLICATION_FLAGS_NONE);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);

    return status;
}

