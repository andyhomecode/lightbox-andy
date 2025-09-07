package src.model;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class PhotoFinder {
    private final String photoDir;

    public PhotoFinder() {
        // Read PHOTODIR from environment variable or default to "./photos"
        this.photoDir = System.getenv().getOrDefault("PHOTODIR", "./photos");
    }

    public List<File> findPhotos() {
        File dir = new File(photoDir);
        if (!dir.exists() || !dir.isDirectory()) {
            return new ArrayList<>();
        }
        // Accept common photo file extensions
        String[] extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"};
        return Arrays.stream(dir.listFiles())
                .filter(File::isFile)
                .filter(f -> {
                    String name = f.getName().toLowerCase();
                    return Arrays.stream(extensions).anyMatch(name::endsWith);
                })
                .sorted((f1, f2) ->
                    f1.getName().compareToIgnoreCase(f2.getName()))
                .collect(Collectors.toList());
    }

    // Simple test routine
    public static void main(String[] args) {
        PhotoFinder finder = new PhotoFinder();
        List<File> photos = finder.findPhotos();
        System.out.println("Found " + photos.size() + " photo(s):");
        for (File photo : photos) {
            System.out.println(photo.getAbsolutePath());
        }
    }
}