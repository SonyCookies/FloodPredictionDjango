# Generated by Django 5.1.2 on 2024-10-23 03:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data_collection', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='weatherdata',
            name='source',
        ),
    ]
