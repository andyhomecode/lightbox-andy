package src.viewer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.File;
import java.awt.image.BufferedImage;
import src.model.PhotoFinder;

import java.text.SimpleDateFormat; // Added missing import
public class Viewer {
    // Handles image and video display, navigation, and prefetching

    private JFrame frame;
    private JLabel label;
    private java.util.List<BufferedImage> images;
    private java.util.List<File> files;
    private int currentIndex = 0;

    public void showImages(java.util.List<BufferedImage> images, java.util.List<File> files) {
        if (images == null || images.isEmpty()) {
            System.err.println("No images to display.");
            return;
        }
        this.images = images;
        this.files = files;
        this.currentIndex = 0;
        showImageAt(currentIndex);
    }

    private void showImageAt(int idx) {
        BufferedImage img = images.get(idx);
        if (frame == null) {
            frame = new JFrame(files.get(idx).getName());
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setUndecorated(true);
            frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
            label = new JLabel();
            label.setHorizontalAlignment(JLabel.CENTER);
            label.setVerticalAlignment(JLabel.CENTER);
            frame.getContentPane().setBackground(Color.BLACK);
            // Use a layered pane for overlay
            JLayeredPane layeredPane = new JLayeredPane();
            layeredPane.setLayout(new OverlayLayout(layeredPane));
            label.setAlignmentX(0.5f);
            label.setAlignmentY(0.5f);
            layeredPane.add(label, JLayeredPane.DEFAULT_LAYER);
            overlayPanel = new OverlayPanel();
            overlayPanel.setOpaque(false);
            overlayPanel.setAlignmentX(0.5f);
            overlayPanel.setAlignmentY(0.5f);
            layeredPane.add(overlayPanel, JLayeredPane.PALETTE_LAYER);
            frame.setContentPane(layeredPane);
            frame.addKeyListener(new KeyAdapter() {
                @Override
                public void keyPressed(KeyEvent e) {
                    if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                        frame.dispose();
                    } else if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
                        nextImage();
                    } else if (e.getKeyCode() == KeyEvent.VK_LEFT) {
                        prevImage();
                    } else if (e.getKeyCode() == KeyEvent.VK_F1) {
                        overlayVisible = !overlayVisible;
                        overlayPanel.setVisible(overlayVisible);
                        overlayPanel.repaint();
                    }
                }
            });
            frame.setVisible(true);
        } else {
            frame.setTitle(files.get(idx).getName());
        }
        // Scale image to fit screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        Image scaled = img.getScaledInstance(screen.width, screen.height, Image.SCALE_FAST);
        label.setIcon(new ImageIcon(scaled));
        if (overlayPanel != null) {
            overlayPanel.repaint();
        }
    }

    private boolean overlayVisible = true;
    private OverlayPanel overlayPanel;

    // Overlay panel for filename and date
    private class OverlayPanel extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (!overlayVisible || files == null || files.isEmpty()) return;
            File file = files.get(currentIndex);
            String filename = file.getName();
            String date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date(file.lastModified()));
            String text = filename + "  |  " + date;
            Graphics2D g2d = (Graphics2D) g.create();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            Font font = new Font("SansSerif", Font.BOLD, 32);
            g2d.setFont(font);
            FontMetrics fm = g2d.getFontMetrics();
            int textWidth = fm.stringWidth(text);
            int textHeight = fm.getHeight();
            int x = (getWidth() - textWidth) / 2;
            int y = getHeight() - textHeight - 40;
            // Draw translucent background
            g2d.setColor(new Color(0, 0, 0, 160));
            g2d.fillRoundRect(x - 20, y - textHeight + 8, textWidth + 40, textHeight + 16, 30, 30);
            // Draw text
            g2d.setColor(Color.WHITE);
            g2d.drawString(text, x, y);
            g2d.dispose();
        }
    }
    private void nextImage() {
        if (images == null || images.isEmpty()) return;
        currentIndex = (currentIndex + 1) % images.size();
        showImageAt(currentIndex);
    }

    private void prevImage() {
        if (images == null || images.isEmpty()) return;
        currentIndex = (currentIndex - 1 + images.size()) % images.size();
        showImageAt(currentIndex);
    }

    // Tester: preload photos with PhotoFinder and allow navigation with arrows
    public static void main(String[] args) {
        PhotoFinder finder = new PhotoFinder();
        java.util.List<BufferedImage> images = finder.preloadPhotos();
        java.util.List<File> files = finder.getPhotoFiles();
        if (images.isEmpty()) {
            System.err.println("No photos found or loaded.");
            return;
        }
        System.out.println("Loaded " + images.size() + " photo(s).");
        new Viewer().showImages(images, files);
    }
}