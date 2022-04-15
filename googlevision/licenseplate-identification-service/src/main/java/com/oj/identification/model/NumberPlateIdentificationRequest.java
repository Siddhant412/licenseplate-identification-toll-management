package com.oj.identification.model;

import lombok.Builder;
import lombok.Data;

import java.io.InputStream;


@Data
@Builder
public class NumberPlateIdentificationRequest {

    private InputStream numberPlateIdentificationData;

}
