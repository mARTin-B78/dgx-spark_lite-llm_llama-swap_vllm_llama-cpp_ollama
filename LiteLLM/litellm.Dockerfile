# litellm.Dockerfile
# We pull from their stable release to ensure local reliability
FROM docker.litellm.ai/berriai/litellm-database:main-stable
# No extra build steps needed, this just mirrors it to your GHCR for your users
