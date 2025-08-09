# Debian linux distribution with python 3.9
FROM --platform=linux/amd64 python:3.9-slim-bookworm

# Set the working directory
WORKDIR /pyspi_project

# Copy the current directory contents into the container
COPY . .

# Update the package index and install essential packages
RUN apt-get update && apt-get install -y build-essential octave

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    python setup.py install

# Make port 80 available to communicate with other containters if needed
# EXPOSE 80

# Run app.py when the container launches
CMD ["python"]
