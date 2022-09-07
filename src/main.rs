use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::{vk, vk::Handle, Device, Entry, Instance};
use clap::Parser;
use std::borrow::Cow;
use std::ffi::{CStr, CString};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_parser)]
    width: Option<u32>,

    #[clap(short, long, value_parser)]
    height: Option<u32>,

    #[clap(short, long)]
    igpu: bool,
}

struct LaunchOptions {
    width: u32,
    height: u32,
    integrated: bool,
}

impl From<&Cli> for LaunchOptions {
    fn from(cli: &Cli) -> Self {
        LaunchOptions {
            width: cli.width.unwrap_or(1280),
            height: cli.height.unwrap_or(720),
            integrated: cli.igpu,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let options = LaunchOptions::from(&cli);

    let renderer = Core::new(&options);
}

struct Core {
    //sdl2
    sdl: sdl2::Sdl,
    sdl_video: sdl2::VideoSubsystem,
    window: sdl2::video::Window,

    //ash
    entry: Entry,
    instance: Instance,
    device: Device,
    surface_loader: Surface,
    swapchain_loader: Swapchain,
    debug_utils_loader: DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,

    //vulkan
    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    present_queue: vk::Queue,

    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_resoultion: vk::Extent2D,

    swapchain: vk::SwapchainKHR,
    present_images: Vec<vk::Image>,
    present_image_views: Vec<vk::ImageView>,

    command_pool: vk::CommandPool,
    draw_command_buffer: vk::CommandBuffer,
    setup_command_buffer: vk::CommandBuffer,

    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,

    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,

    draw_commands_reuse_fence: vk::Fence,
    setup_commands_reuse_fence: vk::Fence,
}

impl Core {
    pub fn new(options: &LaunchOptions) -> Self {
        unsafe {
            let sdl = sdl2::init().expect("Failed to initialize SDL2");
            let sdl_video = sdl.video().expect("Failed to initialize SDL2 Video");

            let package_name = env!("CARGO_PKG_NAME");

            let window = sdl_video
                .window(package_name, options.width, options.height)
                .position_centered()
                .vulkan()
                .build()
                .expect("Failed to create SDL2 Window");

            let entry = Entry::load().expect("Failed to load vulkan libraries");

            let application_name = CString::new("Rustle").unwrap();

            let application_info = vk::ApplicationInfo::builder()
                .application_name(&application_name)
                .application_version(0)
                .engine_name(&application_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 3, 0));

            /* Issue 101398 on rust github
            let errors_example: Vec<_> = [
                CString::new("VK_LAYER_KHRONOS_validation").unwrap()
            ].iter().map(|l| l.as_ptr()).collect();
            */

            let instance_layers = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];

            let instance_layer_names: Vec<_> = instance_layers.iter().map(|l| l.as_ptr()).collect();

            let instance_extensions: Vec<_> = window
                .vulkan_instance_extensions()
                .expect("Failed to query SDL2 window vulkan extensions")
                .iter()
                .map(|&i| CString::new(i).unwrap())
                .collect();

            let mut instance_extension_names: Vec<_> =
                instance_extensions.iter().map(|e| e.as_ptr()).collect();
            instance_extension_names.push(DebugUtils::name().as_ptr());

            let instance_info = vk::InstanceCreateInfo::builder()
                .application_info(&application_info)
                .enabled_extension_names(&instance_extension_names)
                .enabled_layer_names(&instance_layer_names);

            let instance: Instance = entry
                .create_instance(&instance_info, None)
                .expect("Failed to create vulkan instance");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let surface = vk::SurfaceKHR::from_raw(
                window
                    .vulkan_create_surface(instance.handle().as_raw() as usize)
                    .expect("Failed to create SDL surface"),
            );
            let surface_loader = Surface::new(&entry, &instance);

            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failed to find vulkan physical devices");

            let device_desireability: Vec<(Option<_>, _)> = physical_devices
                .iter()
                .map(|device| {
                    let queue_family_index = instance
                        .get_physical_device_queue_family_properties(*device)
                        .iter()
                        .enumerate()
                        .find_map(|(queue_index, info)| {
                            let supports_required =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *device,
                                            queue_index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_required {
                                Some(queue_index)
                            } else {
                                None
                            }
                        });

                    let memory_properties = instance.get_physical_device_memory_properties(*device);

                    let memory = memory_properties
                        .memory_heaps
                        .iter()
                        .find(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                        .unwrap();

                    (
                        queue_family_index,
                        if options.integrated {
                            match instance.get_physical_device_properties(*device).device_type {
                                vk::PhysicalDeviceType::INTEGRATED_GPU => memory.size,
                                _ => 0,
                            }
                        } else {
                            match instance.get_physical_device_properties(*device).device_type {
                                vk::PhysicalDeviceType::DISCRETE_GPU => memory.size * 1000,
                                vk::PhysicalDeviceType::INTEGRATED_GPU => memory.size * 50,
                                _ => memory.size,
                            }
                        },
                    )
                })
                .collect();

            let mut high_score = 0;
            let mut device_index = 0;
            let mut queue_family_index = 0;
            for (index, (queue_family, score)) in device_desireability.iter().enumerate() {
                if let Some(queue_family) = queue_family {
                    if *score > high_score {
                        high_score = *score;
                        device_index = index;
                        queue_family_index = *queue_family;
                    }
                }
            }

            let physical_device = physical_devices[device_index];

            let queue_family_index = queue_family_index as u32;
            let device_extension_names = [Swapchain::name().as_ptr()];

            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };

            let priorities = [1.0];

            let queue_infos = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let device_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_infos))
                .enabled_extension_names(&device_extension_names)
                .enabled_features(&features);

            let device = instance
                .create_device(physical_device, &device_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index, 0);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0];

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap();

            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }

            let surface_resoultion = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: options.width,
                    height: options.height,
                },
                _ => surface_capabilities.current_extent,
            };

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };

            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap();

            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resoultion)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_info, None)
                .unwrap();

            let pool_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let command_pool = device.create_command_pool(&pool_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();

            let setup_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    device.create_image_view(&view_info, None).unwrap()
                })
                .collect();

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);

            let depth_image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(surface_resoultion.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = device.create_image(&depth_image_info, None).unwrap();
            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);

            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .expect("Unable to allocate depth image memory");

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory");

            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

            let draw_commands_reuse_fence = device
                .create_fence(&fence_info, None)
                .expect("Failed to create fence");
            let setup_commands_reuse_fence = device
                .create_fence(&fence_info, None)
                .expect("Failed to create fence");

            record_submit_commandbuffer(
                &device,
                setup_command_buffer,
                setup_commands_reuse_fence,
                present_queue,
                &[],
                &[],
                &[],
                |device, setup_command_buffer| {
                    let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
                        .image(depth_image)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .layer_count(1)
                                .level_count(1)
                                .build(),
                        )
                        .build();

                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    );
                },
            );

            let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(depth_image)
                .format(depth_image_info.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            Core {
                sdl,
                sdl_video,
                window,
                entry,
                instance,
                device,
                surface_loader,
                swapchain_loader,
                debug_utils_loader,
                debug_call_back,
                physical_device,
                memory_properties,
                queue_family_index,
                present_queue,
                surface,
                surface_format,
                surface_resoultion,
                swapchain,
                present_images,
                present_image_views,
                command_pool,
                draw_command_buffer,
                setup_command_buffer,
                depth_image,
                depth_image_view,
                depth_image_memory,
                present_complete_semaphore,
                rendering_complete_semaphore,
                draw_commands_reuse_fence,
                setup_commands_reuse_fence,
            }
        }
    }

    fn main() -> Result<(), String> {
        Ok(())
    }
}

fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop
        .memory_types
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _)| index as _)
}

#[allow(clippy::too_many_arguments)]
fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .wait_for_fences(&[command_buffer_reuse_fence], true, std::u64::MAX)
            .expect("Wait for fence failed");
        device
            .reset_fences(&[command_buffer_reuse_fence])
            .expect("Reuse fence failed");

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset commandbuffer failed");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begine commandbuffer");

        f(device, command_buffer);

        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores)
            .build();

        device
            .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
            .expect("queue submit failed");
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_dat: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:{:?} [{} ({})] : {}",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message
    );

    vk::FALSE
}
