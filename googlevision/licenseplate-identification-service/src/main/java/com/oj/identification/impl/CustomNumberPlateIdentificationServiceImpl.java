package com.oj.identification.impl;

import com.oj.identification.NumberPlateIdentification;
import com.oj.identification.model.NumberPlateIdentificationRequest;
import com.oj.identification.model.NumberPlateIdentificationResponse;
import com.oj.identification.model.ResponseScore;

public class CustomNumberPlateIdentificationServiceImpl implements NumberPlateIdentification {



    @Override
    public NumberPlateIdentificationResponse detect(NumberPlateIdentificationRequest request) {

        return NumberPlateIdentificationResponse.builder()
                .responseScore(ResponseScore.GOOD).build();
    }
}
