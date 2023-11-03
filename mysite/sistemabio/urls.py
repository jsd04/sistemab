
from django.urls import path

from . import views
from . import views2


app_name = "sistemabio"
urlpatterns = [
    path("",views.index, name="home"),
    path("home_admin/",views.home_admin, name="home_admin"),
    path("home_usuario/",views.home_usuario, name="home_usuario"),
    path("ingresar_usuario/",views.ingresar_usuario, name="ingresar_usuario"),
    path("registrar_usuario/",views.registrar_usuario, name="registrar_usuario"),

    path("signup/",views.signup, name="signup"),
    path("signin/",views.signin, name="signin"),
    path("logout/",views.signout,name="logout"),
    path("about/",views.about, name="about"),
    path("principal/",views.principal, name="principal"),
    path("perfil_administrador", views.perfil_administrador, name="perfil_administrador"),
    path("administradores", views.administradores, name="administradores"),

    #Inquilinos
    path("inquilinos/",views.inquilinos, name="inquilinos"),
    path("new_inquilino/",views.new_inquilino, name="new_inquilino"),
    path("search_inquilino/",views.search_inquilino, name="search_inquilino"),
    path("detail_inquilino/<int:usuario_id>/",views.detail_inquilino, name ="detail_inquilino"),
    # path("detail_inquilino2/<int:usuario_id>/<int:sesion_idu>/",views.detail_inquilino2, name ="detail_inquilino2"),
    path("delete_inquilino/<int:inquilino_id>/", views.delete_inquilino, name="delete_inquilino"),
    path("edit_inquilino/<int:usuario_id>/", views.edit_inquilino, name="edit_inquilino"),

    #Biométricos
    path("new_biometricos/", views2.new_biometricos, name="new_biometricos"),
    path("new_biometrico/<int:usuario_id>/", views.new_biometrico, name="new_biometrico"),
    #Datos biométricos
    path("facial/<int:usuario_id>/",views2.facial, name="facial"),
    path("voz3/<int:usuario_id>/", views2.voz3,name="voz3"),
    path("huella/<int:usuario_id>/", views2.huella,name="huella"),

    path("facial_usuario/",views2.facial_usuario, name="facial_usuario"),
    path("voz_usuario/", views2.voz_usuario,name="voz_usuario"),
    path("huella_usuario/", views2.huella_usuario,name="huella_usuario"),

    path("accediste/", views2.accediste,name="accediste"),


    
] 
   