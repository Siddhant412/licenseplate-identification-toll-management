package com.oj.configuration;

import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.services.textract.AmazonTextract;
import com.amazonaws.services.textract.AmazonTextractClientBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AWSTextractConfiguration {

    @Bean
    public AmazonTextract getTextractClient(){
        AmazonTextract amazonTextract = AmazonTextractClientBuilder.
                standard().withCredentials(getCredentials()).build();
        return amazonTextract;
    }

    @Bean
    public AWSCredentialsProvider getCredentials(){
        AWSCredentialsProvider awsCredentialsProvider = new AWSStaticCredentialsProvider(
            new BasicAWSCredentials("access","")
        );
                return awsCredentialsProvider;
    }



}
