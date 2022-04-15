package com.oj.identification.model;

import lombok.Data;
import lombok.Getter;

@Getter
public enum ResponseScore {

    VERY_GOOD(10), GOOD(8), NOT_GOOD(5);

    private Integer responseScore;
     ResponseScore(Integer score){
         this.responseScore = score;
    }
}
