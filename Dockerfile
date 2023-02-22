FROM python:3.9-buster

# Octave install
RUN apt-get update \
    && apt-get install -y octave less python-scipy libblas-dev liblapack-dev gfortran redis-server \
    && apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Java install
RUN apt-get update && \
    apt-get install -y default-jdk && \ rm -rf /var/lib/apt/lists/

# Set the working directory
WORKDIR /pyspi_project

# Copy the current directory contents into the container at /app
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
#ENV NAME Wold

# Run app.py when the container launches
CMD ["python3", "setup.py"]
