// AWS access
provider "aws" {
  profile = "default"
  region  = "us-east-2"
}

// Resources
resource "aws_s3_bucket" "main_bucket" {
  bucket = "alnn-main-bucket-8nyb87yn8"
  acl    = "public-read"
}

resource "aws_s3_bucket" "animations_bucket" {
  bucket = "alnn-animations-bucket-8nyb87yn8"
  acl    = "public-read"
}

resource "aws_s3_bucket" "zipfiles_bucket" {
  bucket = "alnn-zipfiles-bucket-8nyb87yn8"
  acl    = "public-read"
}