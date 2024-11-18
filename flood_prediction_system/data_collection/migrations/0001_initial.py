# Generated by Django 5.1.2 on 2024-10-23 02:44

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='WeatherData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('datetime', models.DateTimeField()),
                ('tempmax', models.FloatField()),
                ('tempmin', models.FloatField()),
                ('temp', models.FloatField()),
                ('feelslikemax', models.FloatField()),
                ('feelslikemin', models.FloatField()),
                ('feelslike', models.FloatField()),
                ('dew', models.FloatField()),
                ('humidity', models.FloatField()),
                ('precip', models.FloatField()),
                ('precipprob', models.FloatField()),
                ('precipcover', models.FloatField()),
                ('preciptype', models.CharField(max_length=100)),
                ('snow', models.FloatField()),
                ('snowdepth', models.FloatField()),
                ('windgust', models.FloatField()),
                ('windspeed', models.FloatField()),
                ('winddir', models.FloatField()),
                ('pressure', models.FloatField()),
                ('cloudcover', models.FloatField()),
                ('visibility', models.FloatField()),
                ('solarradiation', models.FloatField()),
                ('solarenergy', models.FloatField()),
                ('uvindex', models.FloatField()),
                ('sunrise', models.TimeField(null=True)),
                ('sunset', models.TimeField(null=True)),
                ('moonphase', models.FloatField(null=True)),
                ('conditions', models.CharField(max_length=100, null=True)),
                ('description', models.TextField(null=True)),
                ('icon', models.CharField(max_length=100, null=True)),
                ('stations', models.TextField(null=True)),
                ('source', models.CharField(max_length=100, null=True)),
            ],
        ),
    ]
