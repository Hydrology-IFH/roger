# first stage
# set base image (host OS)
FROM python:3.9 AS builder

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --user -r requirements.txt

# second unnamed stage
FROM python:3.9-slim
WORKDIR /roger

# copy only the dependencies installation from the 1st stage image
COPY --from=builder /root/.local /root/.local
# Copy the source code into the container.
COPY . .

# update PATH environment variable
ENV PATH=/root/.local:$PATH
# update PYTHONPATH environment variable to use the roger source code
ENV PYTHONPATH=/roger

CMD ["python", "/roger/examples/plot_scale/svat_tutorial/svat.py"]