plugins {
    id("com.android.application")
}

android {
    namespace = "kr.re.etri.fdm.smartlens"
    compileSdk = 36

    defaultConfig {
        applicationId = "kr.re.etri.fdm.smartlens"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {

        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    // With AGP 9's built-in Kotlin, kotlinOptions is removed; the Kotlin jvmTarget
    // defaults to android.compileOptions.targetCompatibility (JavaVersion.VERSION_17).

    buildFeatures {
        viewBinding = true
    }
}

dependencies {

    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.13.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("androidx.test:monitor:1.7.2")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")

    val cameraxVersion = "1.5.3"
    implementation("androidx.camera:camera-camera2:${cameraxVersion}")
    implementation("androidx.camera:camera-lifecycle:${cameraxVersion}")
    implementation("androidx.camera:camera-view:${cameraxVersion}")

    // LiteRT (successor to org.tensorflow:tensorflow-lite; keeps the org.tensorflow.lite.* API).
    // Replaces the old org.tensorflow:tensorflow-lite* artifacts, which duplicated LiteRT's
    // classes under AGP 9. Coherent 1.4.2 set: litert-support -> litert core; litert-gpu -> api.
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
    implementation("com.google.ai.edge.litert:litert-support:1.4.2")

    // Gson for JSON serialization
    implementation("com.google.code.gson:gson:2.14.0")

    // RecyclerView for history list
    implementation("androidx.recyclerview:recyclerview:1.4.0")
}