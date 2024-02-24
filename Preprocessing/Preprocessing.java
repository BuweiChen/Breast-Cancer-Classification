import com.idrsolutions.image.JDeli;
import com.idrsolutions.image.process.ImageProcessingOperations;
import com.pixelmed.dicom.*;
import com.pixelmed.display.SourceImage;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.geom.AffineTransform;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.nio.Buffer;
import java.util.Scanner;

public class Preprocessing {
    static AttributeList list = new AttributeList();
    public static void main(String[] args) throws Exception {
        Scanner in = new Scanner(new File("D:/calc_case_description_train_set.csv"));
        String[][] usefulInfoCalcTrain = parseCSVCalc(in, 1546);

        for (int i = 309; i < 1546; i++)
        {
            String dcm = getLocationOfDcm(usefulInfoCalcTrain, i);
            String tiff = getLocationOfTiffTrain(usefulInfoCalcTrain, i);
            try {
                dcmToBuffered(dcm);
            }
            catch (Exception e) {
                usefulInfoCalcTrain[i][3] = usefulInfoCalcTrain[i][3].equals("1") ? "2" : "1";
                dcm = getLocationOfDcm(usefulInfoCalcTrain, i);
            }
            BufferedImage image = dcmToBuffered(dcm);
            create4Rotations(image, tiff, 0);
            System.out.println("progress1: " + (i*100/1546) + "%");
        }

        in = new Scanner(new File("D:/calc_case_description_test_set.csv"));
        String[][] usefulInfoCalcTest = parseCSVCalc(in, 326);
        for (int i = 0; i < 326; i++)
        {
            String dcm = getLocationOfDcm(usefulInfoCalcTest, i);
            String tiff = getLocationOfTiffTest(usefulInfoCalcTest, i);
            try {
                dcmToBuffered(dcm);
            }
            catch (Exception e) {
                usefulInfoCalcTest[i][3] = usefulInfoCalcTest[i][3].equals("1") ? "2" : "1";;
                dcm = getLocationOfDcm(usefulInfoCalcTest, i);
            }
            BufferedImage image = dcmToBuffered(dcm);
            create4Rotations(image, tiff, 0);
            System.out.println("progress2: " + (i*100/326) + "%");
        }


    }

    private static String getLocationOfDcm(String[][] usefulInfo, int row)
    {
        String dcmLocation = usefulInfo[row][2];
        File file = new File (dcmLocation);
        dcmLocation += "/" + file.list()[0];
        file = new File (dcmLocation);
        dcmLocation += "/" + file.list()[0] + "/1-" + usefulInfo[row][3] + ".dcm";
        return dcmLocation;
    }

    private static BufferedImage dcmToBuffered(String location) throws IOException, DicomException {
        list.read(location);
        OtherWordAttribute pixelAttribute = (OtherWordAttribute)(list.get(TagFromName.PixelData));
        short[] data = pixelAttribute.getShortValues();
        SourceImage img = new SourceImage(list);
        int width = img.getWidth();
        int height = img.getHeight();
        DataBuffer dataBuffer = new DataBufferUShort(data, data.length);
        int stride = 1;
        WritableRaster raster = Raster.createInterleavedRaster(dataBuffer, width, height, width * stride, stride, new int[] {0}, null);
        ColorModel colorModel = new ComponentColorModel(ColorSpace.getInstance(ColorSpace.CS_GRAY), false, false, Transparency.OPAQUE, DataBuffer.TYPE_USHORT);
        BufferedImage converted = new BufferedImage(colorModel, raster, colorModel.isAlphaPremultiplied(), null);
        return converted;
    }

    private static String getLocationOfTiffTrain(String[][] usefulInfo, int row) {
        String tiffLocation = "D:/CBIS-DDSM-train/";
        if (Integer.parseInt(usefulInfo[row][0]) == 0)
        {
            tiffLocation += "benign/";
        }
        else
            tiffLocation += "malignant/";

        tiffLocation += usefulInfo[row][2].substring(13) + "_lv" +usefulInfo[row][1];
        return tiffLocation;
    }

    private static String getLocationOfTiffTest(String[][] usefulInfo, int row) {
        String tiffLocation = "D:/CBIS-DDSM-test/";
        if (Integer.parseInt(usefulInfo[row][0]) == 0)
        {
            tiffLocation += "benign/";
        }
        else
            tiffLocation += "malignant/";

        tiffLocation += usefulInfo[row][2].substring(13) + "lv" +usefulInfo[row][1];
        return tiffLocation;
    }

    private static void create4Rotations(BufferedImage image, String location, int i) throws IOException {
        if (i < 1) {
            File file = new File(location + "_r" + i + ".tif");
            ImageIO.write(image, "TIFF", file);
            create4Rotations(image, location, i+1);
        }
        else if (i < 4) {
            double rads = Math.toRadians(90);
            double sin = Math.abs(Math.sin(rads));
            double cos = Math.abs(Math.cos(rads));
            int w = (int) Math.floor(image.getWidth() * cos + image.getHeight() * sin);
            int h = (int) Math.floor(image.getHeight() * cos + image.getWidth() * sin);
            BufferedImage rotatedImage = new BufferedImage(w, h, image.getType());
            AffineTransform at = new AffineTransform();
            at.translate(w / 2, h / 2);
            at.rotate(rads, 0, 0);
            at.translate(-image.getWidth() / 2, -image.getHeight() / 2);
            final AffineTransformOp rotateOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
            rotateOp.filter(image, rotatedImage);
            File file = new File(location + "_r" + i + ".tif");
            ImageIO.write(rotatedImage, "TIFF", file);
            create4Rotations(rotatedImage, location, i+1);
        }
    }

    private static String getFileLocationCalc(String name)
    {
        String fileName = "D:/CBIS-DDSM/";
        int i = name.indexOf("/");
        fileName += name.substring(1, i);
        return fileName;
    }

    private static String getFileLocationMass(String name)
    {
        String fileName = "D:/CBIS-DDSM/";
        int i = name.indexOf("/");
        fileName += name.substring(0, i);
        return fileName;
    }

    private static String[][] parseCSVCalc(Scanner in, int rol){
        String[][] usefulInfo = new String[rol][4];
        in.nextLine();
        for (int i = 0; i < rol; i++) {
            String[] ln = in.nextLine().split(",");
            if (ln[9].substring(0, 1).equals("B") || ln[9].substring(0, 1).equals("b"))
                usefulInfo[i][0] = "0";
            else
                usefulInfo[i][0] = "1";
            usefulInfo[i][1] = ln[10];
            usefulInfo[i][2] = getFileLocationCalc(ln[12]);
            usefulInfo[i][3] = ln[12].substring(ln[12].length() - 5, ln[12].length() - 4);
            in.nextLine();
        }
        return usefulInfo;
    }

    private static String[][] parseCSVMass(Scanner in, int rol){
        String[][] usefulInfo = new String[rol][4];
        in.nextLine();
        for (int i = 0; i < rol; i++) {
            String[] ln = in.nextLine().split(",");
            if (ln[9].substring(0, 1).equals("B") || ln[9].substring(0, 1).equals("b"))
                usefulInfo[i][0] = "0";
            else
                usefulInfo[i][0] = "1";
            usefulInfo[i][1] = ln[10];
            usefulInfo[i][2] = getFileLocationMass(ln[12]);
            usefulInfo[i][3] = ln[12].substring(ln[12].length() - 5, ln[12].length() - 4);
            in.nextLine();
        }
        return usefulInfo;
    }
}
