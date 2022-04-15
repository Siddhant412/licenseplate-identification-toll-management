package com.oj.job;

import com.oj.event.Event;
import com.oj.event.EventName;
import com.oj.event.EventStatus;
import com.oj.identification.NumberPlateIdentification;
import com.oj.identification.model.NumberPlateIdentificationRequest;
import com.oj.identification.model.ResponseScore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jms.annotation.JmsListener;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.Collection;

@Service
public class JobReader {

    @Autowired
    Collection<NumberPlateIdentification> numberPlateIdentifications;

    @JmsListener(destination = "thumbnail_requests")
    public void processEvent(Event event) {

        //assuming message is coming from capture service and status is success to proceed with processing
        if (EventName.CAPTURE.equals(event.getName()) && EventStatus.SUCCESS.equals(event.getStatus())) {
            InputStream inputStream = event.getCaptureLocation().getClass().getResourceAsStream(""); // this will be replacted by S3 to read the object from a location
            NumberPlateIdentificationRequest numberPlateIdentificationRequest = NumberPlateIdentificationRequest.builder()
                    .numberPlateIdentificationData(inputStream)
                    .build();

            for(NumberPlateIdentification numberPlateIdentification: numberPlateIdentifications){
                ResponseScore responseScore = numberPlateIdentification.detect(numberPlateIdentificationRequest).getResponseScore();
                if(responseScore.getResponseScore() > 7){
                    break;// when better score is achieved which is defined as 5 as of now then program will terminate otherwise fallback logic will execute
                }
                else{
                    //manual workflow trigger

                    // manual workflow queue with Event as Manual
                }
            }

        }
    }

}
