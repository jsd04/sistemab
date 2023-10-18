# Generated by Django 4.2.2 on 2023-10-18 20:57

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Usuario',
            fields=[
                ('id_usuario', models.AutoField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=30)),
                ('ap_paterno', models.CharField(max_length=15)),
                ('ap_materno', models.CharField(max_length=15)),
                ('curp', models.CharField(max_length=30)),
                ('piso', models.IntegerField()),
                ('departamento', models.IntegerField()),
                ('telefono', models.CharField(max_length=15)),
                ('correo', models.EmailField(max_length=254)),
                ('fecha_nac', models.DateField()),
                ('fecha_creado', models.DateTimeField(auto_now_add=True)),
                ('fecha_actualizado', models.DateTimeField(auto_now=True)),
                ('id_perfil', models.CharField(choices=[('1', 'Administrador'), ('2', 'Usuario Inquilino'), ('3', 'Usuario Visitante'), ('4', 'Usuario Trabajador')], default='2', max_length=2)),
                ('id_status', models.CharField(choices=[('1', 'ALTA'), ('2', 'BAJA'), ('3', 'PENDIENTE'), ('4', 'ALTA/COMPLETO')], default='3', max_length=2)),
            ],
        ),
        migrations.CreateModel(
            name='Sesion',
            fields=[
                ('id_sesion', models.AutoField(primary_key=True, serialize=False)),
                ('id_tipo_sesion', models.CharField(choices=[('1', 'FACIAL'), ('2', 'VOZ'), ('3', 'HUELLA')], max_length=2)),
                ('completado', models.BooleanField(default=False)),
                ('dato', models.BinaryField(editable=True)),
                ('fecha_creacion', models.DateTimeField(auto_now_add=True)),
                ('fecha_actualizacion', models.DateTimeField(auto_now=True)),
                ('id_usuario', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='sistemabio.usuario')),
            ],
        ),
    ]
