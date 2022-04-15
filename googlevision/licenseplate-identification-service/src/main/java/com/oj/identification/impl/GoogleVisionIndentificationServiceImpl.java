package com.oj.identification.impl;

import com.google.cloud.translate.v3.TranslateTextRequest;
import com.google.cloud.translate.v3.TranslateTextResponse;
import com.google.cloud.translate.v3.Translation;
import com.google.cloud.translate.v3.TranslationServiceClient;
import com.google.cloud.vision.v1.*;
import com.google.protobuf.ByteString;
import com.oj.identification.NumberPlateIdentification;
import com.oj.identification.model.NumberPlateIdentificationRequest;
import com.oj.identification.model.NumberPlateIdentificationResponse;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import static java.lang.System.out;

public class GoogleVisionIndentificationServiceImpl implements NumberPlateIdentification {


    @Override
    public NumberPlateIdentificationResponse detect(NumberPlateIdentificationRequest identificationRequest) {

        List<AnnotateImageRequest> requests = new ArrayList<>();

        ByteString imgBytes = null;
        String text = null;
        try {
            imgBytes = ByteString.readFrom(new FileInputStream("C:\\Omkar\\Deep Blue\\Test_Data\\images_1581599221866_marathi_number_plate.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        Image img = Image.newBuilder().setContent(imgBytes).build();
        Feature feat = Feature.newBuilder().setType(Feature.Type.TEXT_DETECTION).build();
        AnnotateImageRequest request =
                AnnotateImageRequest.newBuilder().addFeatures(feat).setImage(img).setImageContext(ImageContext.newBuilder().addLanguageHints("mr")).build();
        requests.add(request);

        try (ImageAnnotatorClient client = ImageAnnotatorClient.create()) {
            BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
            List<AnnotateImageResponse> responses = response.getResponsesList();

            for (AnnotateImageResponse res : responses) {
                if (res.hasError()) {
                    out.printf("Error: %s\n", res.getError().getMessage());
                    break;
                }

                // For full list of available annotations, see http://g.co/cloud/vision/docs
                for (EntityAnnotation annotation : res.getTextAnnotationsList()) {
                    out.printf("Text: %s\n", annotation.getDescription());
                    text = annotation.getDescription();
                    out.printf("Position : %s\n", annotation.getBoundingPoly());
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Initialize client that will be used to send requests. This client only needs to be created
        // once, and can be reused for multiple requests. After completing all of your requests, call
        // the "close" method on the client to safely clean up any remaining background resources.
        try (TranslationServiceClient client = TranslationServiceClient.create()) {
            // Supported Locations: `global`, [glossary location], or [model location]
            // Glossaries must be hosted in `us-central1`
            // Custom Models must use the same location as your model. (us-central1)
            LocationName parent = LocationName.of("psychic-glider-341015", "global");

            // Supported Mime Types: https://cloud.google.com/translate/docs/supported-formats
            TranslateTextRequest translateTextRequest =
                    TranslateTextRequest.newBuilder()
                            .setParent(parent.toString())
                            .setMimeType("text/plain")
                            .setTargetLanguageCode("en")
                            .addContents(text)
                            .build();

            TranslateTextResponse response = client.translateText(translateTextRequest);

            // Display the translation for each input text provided
            for (Translation translation : response.getTranslationsList()) {
                System.out.printf("Translated text: %s\n", translation.getTranslatedText());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


        return null;
    }

    public static void main(String[] args) {
        GoogleVisionIndentificationServiceImpl googleVisionIndentificationService = new GoogleVisionIndentificationServiceImpl();
        googleVisionIndentificationService.detect(null);

        /*String locationId = System.getenv().get("locationId");

        String vehicleId = UUID.randomUUID().toString();

        String crop = "cropLocation";

        out.println("creating a database record for a vehicle");

        String croppedImage = "croppedImage";

        out.println("uploaded file at " +"gs:" + croppedImage );


        out.println("read the database record");

        out.println("OCR " + "Translation");

        String licensePlate = "MH 02 CXX UUU";

        out.println("update database record with license plate no with status as SUCCESS");

        out.println("if not found then update status as MANUAL_INTERVENTION_REQUIRED");


        out.println("if status SUCCESS then amount deduction service will read the record");

        out.println("go to the website which has owner information");

        String vehicleMake, model, engineNumber, ccm, owner = "";

        out.println("call matrix table");

        out.println("deduction Amount");

        out.println("");*/




    }
}
