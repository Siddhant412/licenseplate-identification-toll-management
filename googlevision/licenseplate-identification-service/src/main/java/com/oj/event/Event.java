package com.oj.event;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class Event {

    private String captureLocation;

    private LocalDateTime createdAt;

    private EventStatus status;

    private EventName name;

    private String evidenceCaptureLocation;

}


