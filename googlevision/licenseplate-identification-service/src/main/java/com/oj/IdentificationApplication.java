package com.oj;


import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class IdentificationApplication implements CommandLineRunner {

    public static void main(String[] args) {
        SpringApplication.run(IdentificationApplication.class,args);
    }


    @Override
    public void run(String... args) throws Exception {
        while(true){
            //this infite loop is to make sure main thread is always running
        }
    }
}
