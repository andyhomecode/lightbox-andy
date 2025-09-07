package src.model;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

public class PhotoFinder {
    private List<File> photoFiles = new ArrayList<>();

    public List<BufferedImage> preloadPhotos() {
        photoFiles.clear();
        List<BufferedImage> images = new ArrayList<>();
        File dir = new File("photos"); // or your image directory
        if (!dir.exists() || !dir.isDirectory()) {
            System.err.println("Photo directory not found: " + dir.getAbsolutePath());
            return images;
        }

        // Get screen size once for prescaling
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int screenW = screen.width;
        int screenH = screen.height;

        File[] files = dir.listFiles((d, name) -> {
            String n = name.toLowerCase();
            return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png") || n.endsWith(".bmp") || n.endsWith(".gif");
        });

        if (files == null) return images;

        for (File file : files) {
            try {
                BufferedImage img = ImageIO.read(file);
                if (img == null) continue;
                // Prescale using fast scaling
                Image scaled = img.getScaledInstance(screenW, screenH, Image.SCALE_FAST);
                BufferedImage prescaled = new BufferedImage(screenW, screenH, BufferedImage.TYPE_INT_ARGB);
                Graphics2D g2d = prescaled.createGraphics();
                g2d.drawImage(scaled, 0, 0, null);
                g2d.dispose();
                images.add(prescaled);
                photoFiles.add(file);
            } catch (Exception e) {
                System.err.println("Failed to load: " + file.getName());
            }
        }
        return images;
    }

    public List<File> getPhotoFiles() {
        return photoFiles;
    }

    // Simple test routine
    public static void main(String[] args) {
        PhotoFinder finder = new PhotoFinder();
        List<BufferedImage> images = finder.preloadPhotos();
        System.out.println("Found and loaded " + images.size() + " photo(s).");
        List<File> files = finder.getPhotoFiles();
        for (int i = 0; i < files.size(); i++) {
            System.out.println(files.get(i).getAbsolutePath() + (images.get(i) != null ? " [OK]" : " [FAILED]"));
        }
    }
}