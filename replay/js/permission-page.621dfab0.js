(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["permission-page"],{3252:function(e,t,n){"use strict";n.r(t);var a=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{staticClass:"app-container"},[n("switch-roles",{on:{change:e.handleRolesChange}})],1)},s=[],c=n("d4ec"),i=n("bee2"),o=n("262e"),r=n("2caf"),l=n("9ab4"),u=n("1b40"),b=n("8ee8"),h=function(e){Object(o["a"])(n,e);var t=Object(r["a"])(n);function n(){return Object(c["a"])(this,n),t.apply(this,arguments)}return Object(i["a"])(n,[{key:"handleRolesChange",value:function(){this.$router.push({path:"/permission/index?"+ +new Date}).catch((function(e){console.warn(e)}))}}]),n}(u["c"]);h=Object(l["a"])([Object(u["a"])({name:"PagePermission",components:{SwitchRoles:b["a"]}})],h);var p=h,f=p,d=n("2877"),v=Object(d["a"])(f,a,s,!1,null,null,null);t["default"]=v.exports},"8ee8":function(e,t,n){"use strict";var a=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",[n("div",{staticStyle:{"margin-bottom":"15px"}},[e._v(" "+e._s(e.$t("permission.roles"))+"： "+e._s(e.roles)+" ")]),e._v(" "+e._s(e.$t("permission.switchRoles"))+"： "),n("el-radio-group",{model:{value:e.switchRoles,callback:function(t){e.switchRoles=t},expression:"switchRoles"}},[n("el-radio-button",{attrs:{label:"editor"}}),n("el-radio-button",{attrs:{label:"admin"}})],1)],1)},s=[],c=n("d4ec"),i=n("bee2"),o=n("262e"),r=n("2caf"),l=n("9ab4"),u=n("1b40"),b=n("9dba"),h=function(e){Object(o["a"])(n,e);var t=Object(r["a"])(n);function n(){return Object(c["a"])(this,n),t.apply(this,arguments)}return Object(i["a"])(n,[{key:"roles",get:function(){return b["a"].roles}},{key:"switchRoles",get:function(){return this.roles[0]},set:function(e){var t=this;b["a"].ChangeRoles(e).then((function(){t.$emit("change")}))}}]),n}(u["c"]);h=Object(l["a"])([Object(u["a"])({name:"SwitchRoles"})],h);var p=h,f=p,d=n("2877"),v=Object(d["a"])(f,a,s,!1,null,null,null);t["a"]=v.exports}}]);