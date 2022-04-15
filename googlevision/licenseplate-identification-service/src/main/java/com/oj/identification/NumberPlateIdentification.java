package com.oj.identification;

import com.oj.identification.model.NumberPlateIdentificationRequest;
import com.oj.identification.model.NumberPlateIdentificationResponse;

public interface NumberPlateIdentification {

    public NumberPlateIdentificationResponse detect(NumberPlateIdentificationRequest request);

}
