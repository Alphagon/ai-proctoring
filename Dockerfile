# Use an official Python image as a base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Create a new environment named 'proctor' with Python 3.7.0
RUN conda create -n proctor python=3.7.0 -y

# Activate the 'proctor' environment and install required packages
# Install dlib using conda-forge channel
RUN /bin/bash -c "source activate proctor && conda install -c conda-forge dlib -y"

# Copy the core requirements file into the container
COPY requirements/core.txt /app/requirements/core.txt

# Install the core Python dependencies
RUN /bin/bash -c "source activate proctor && pip install --no-cache-dir -r /app/requirements/core.txt"

# Copy the dev requirements file into the container
COPY requirements/dev.txt /app/requirements/dev.txt

# Install the development Python dependencies
RUN /bin/bash -c "source activate proctor && pip install --no-cache-dir -r /app/requirements/dev.txt"

# Copy the API requirements file into the container
COPY requirements/api.txt /app/requirements/api.txt

# Install the API-specific Python dependencies
RUN /bin/bash -c "source activate proctor && pip install --no-cache-dir -r /app/requirements/api.txt"

# Copy the source code into the container
COPY src /app/src

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the application
CMD ["/bin/bash", "-c", "source activate proctor && uvicorn src.main:app --host 0.0.0.0 --port 8000"]
