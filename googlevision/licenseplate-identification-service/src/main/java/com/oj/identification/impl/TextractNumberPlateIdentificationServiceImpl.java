package com.oj.identification.impl;

import com.amazonaws.services.textract.AmazonTextract;
import com.amazonaws.services.textract.model.*;
import com.oj.identification.NumberPlateIdentification;
import com.oj.identification.model.NumberPlateIdentificationRequest;
import com.oj.identification.model.NumberPlateIdentificationResponse;
import com.oj.identification.model.ResponseScore;
import org.springframework.beans.factory.annotation.Autowired;

public class TextractNumberPlateIdentificationServiceImpl implements NumberPlateIdentification {

    @Autowired
    private AmazonTextract amazonTextract;


    @Override
    public NumberPlateIdentificationResponse detect(NumberPlateIdentificationRequest request) {

        S3Object s3Object = new S3Object().withBucket("").withName("");

        Document document = new Document().withS3Object(s3Object);

        DetectDocumentTextRequest detectDocumentTextRequest = new DetectDocumentTextRequest().withDocument(document);

        DetectDocumentTextResult detectDocumentTextResult = amazonTextract.detectDocumentText(detectDocumentTextRequest);

        for(Block block: detectDocumentTextResult.getBlocks()){
            block.getText();
        }

        return NumberPlateIdentificationResponse.builder()
                .responseScore(ResponseScore.VERY_GOOD).build();
    }
}
