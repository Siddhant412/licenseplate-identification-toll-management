package com.oj.identification.model;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class NumberPlateIdentificationResponse {

    private ResponseScore responseScore;

    private Evidence evidence;


}
