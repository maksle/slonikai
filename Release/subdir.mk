################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../bb.cpp \
../magics.cpp \
../main.cpp \
../material.cpp \
../movecalc.cpp \
../movegen.cpp \
../position.cpp \
../search.cpp \
../template.cpp \
../zobrist.cpp 

OBJS += \
./bb.o \
./magics.o \
./main.o \
./material.o \
./movecalc.o \
./movegen.o \
./position.o \
./search.o \
./template.o \
./zobrist.o 

CPP_DEPS += \
./bb.d \
./magics.d \
./main.d \
./material.d \
./movecalc.d \
./movegen.d \
./position.d \
./search.d \
./template.d \
./zobrist.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


