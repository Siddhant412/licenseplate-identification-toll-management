package com.oj.configuration;

import com.oj.identification.NumberPlateIdentification;
import com.oj.identification.impl.CustomNumberPlateIdentificationServiceImpl;
import com.oj.identification.impl.TextractNumberPlateIdentificationServiceImpl;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Collection;
import java.util.LinkedHashSet;

@Configuration
public class AppConfiguration {

    @Bean
    public Collection<NumberPlateIdentification> initializeNumberPlateIdentifications(){
        Collection<NumberPlateIdentification> numberPlateIdentifications = new LinkedHashSet<>();
        numberPlateIdentifications.add(initializeTextractIdentification());
        numberPlateIdentifications.add(initializeCustomIdentification());
        return numberPlateIdentifications;
    }

    @Bean
    public TextractNumberPlateIdentificationServiceImpl initializeTextractIdentification(){
        return new TextractNumberPlateIdentificationServiceImpl();
    }
    @Bean
    public CustomNumberPlateIdentificationServiceImpl initializeCustomIdentification(){
        return new CustomNumberPlateIdentificationServiceImpl();
    }


}
