from django.db import models

# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=20, unique=True)
    email = models.EmailField(unique=True)
    # face_data_file = models.FileField(upload_to='face_data/')
    timestamp = models.DateTimeField(auto_now_add=True, blank = True)

    def __str__(self):
        return self.username
