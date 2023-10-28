#![feature(seek_stream_len)]
extern crate rusty_ggml as ggml;

use std::{env, mem};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{bail, Result};

use ggml::context::{GContextBuilder, GGraph};
use ggml::gguf::GGgufContext;
use ggml::gtensor::GMulMat;
use ggml::prelude::{GOpPool, GTensor1, GTensor2, GTensor4, GType, ScratchBuffer};
use rusty_ggml::prelude::GContext;

macro_rules! time {
        ($a:ident($($b:tt)*))=>{
            {
                    use std::time::Instant;

                    let start = Instant::now();
                    let result = $a($($b)*);
                    let end = start.elapsed();
                    eprintln!("[{}:{}] {} took {:?}", file!(), line!(), stringify!($a($($b)*)), end);
                    result
            }
        };
}

struct MnistModel {
    pub conv2d_1_kernel: GTensor4,
    pub conv2d_1_bias: GTensor4,
    pub conv2d_2_kernel: GTensor4,
    pub conv2d_2_bias: GTensor4,
    pub dense_weight: GTensor2,
    pub dense_bias: GTensor1,
    pub ctx: GContext,
}

fn mnist_model_load(model_path: String) -> Result<MnistModel> {
    let gguf_context: GGgufContext = GGgufContext::init_from_file(model_path.as_str(), false)?;
    let mut ctx: GContext = gguf_context.get_ggml_context();
    ctx.register_scratch_buffer(ScratchBuffer::new(4096));

    Ok(MnistModel {
        conv2d_1_kernel: ctx.get_tensor("kernel1")?,
        conv2d_1_bias: ctx.get_tensor("bias1")?,
        conv2d_2_kernel: ctx.get_tensor("kernel2")?,
        conv2d_2_bias: ctx.get_tensor("bias2")?,
        dense_weight: ctx.get_tensor("dense_w")?,
        dense_bias: ctx.get_tensor("dense_b")?,
        ctx,
    })
}

fn mnist_eval(model: &MnistModel, n_threads: usize, digit: &[f32], fname_cgraph: Option<&str>) -> Result<i32> {
    let buf_size = 100000 * std::mem::size_of::<f32>() * 4;
    let ctx0 = GContextBuilder::new()
        .mem_size(buf_size)
        .no_alloc(false)
        .build()?;
    let mut gf = GGraph::new(n_threads);

    let mut input = ctx0.tensor::<4>(GType::F32, [28, 28, 1, 1])?;
    input.populate_f32(digit);
    input.set_name("input");

    let mut cur = model.conv2d_1_kernel.conv_2d(input, 1, 1, 0, 0, 1, 1);
    cur = cur.add(&model.conv2d_1_bias);
    cur = cur.relu();

    // Output shape after Conv2D: (26 26 32 1)
    cur = cur.pool_2d(GOpPool::POOL_MAX, 2, 2, 2, 2, 0, 0);
    // Output shape after MaxPooling2D: (13 13 32 1)
    cur = model.conv2d_2_kernel.conv_2d(cur, 1, 1, 0, 0, 1, 1);
    cur = cur.add(&model.conv2d_2_bias);
    cur = cur.relu();

    // Output shape after Conv2D: (11 11 64 1)
    cur = cur.pool_2d(GOpPool::POOL_MAX, 2, 2, 2, 2, 0, 0);
    // Output shape after MaxPooling2D: (5 5 64 1)
    cur = cur.permute([1, 2, 0, 3]).cont();
    // Output shape after permute: (64 5 5 1)
    let cur = cur.reshape([1600, 1]);
    // Final Dense layer
    let cur = model.dense_weight.mul_mat(&cur); //.add(&model.dense_bias)
    dbg!(cur.shape());
    let mut probs = cur.soft_max();
    probs.set_name("probs");

    gf.build_forward_expand(&probs)?;
    ctx0.compute(&mut gf)?;

    gf.print();
    gf.dump_dot(None, "mnist-cnn.dot");

    if let Some(fname_cgraph) = fname_cgraph {
        // export the compute graph for later use
        // see the "mnist-cpu" example
        gf.export(fname_cgraph);

        println!("exported compute graph to '{}'", fname_cgraph);
    }

    // argmax of probs.data
    let prediction = unsafe {
        probs.with_data(|d| {
            let index_of_max: Option<usize> = mem::transmute::<&[u8], &[f32]>(d)
                .iter()
                .enumerate()
                .max_by(|(_, &a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index);
            index_of_max.unwrap()
        })
    }.map(|x| x as i32);

    return prediction;
}

pub fn main() -> Result<()> {
    let (model_file, test_set_file) = match (env::args().nth(1), env::args().nth(2)) {
        (Some(mf), Some(tsf)) => (mf, tsf),
        _ => {
            bail!(format!("Usage: {} models/mnist/mnist-cnn-model.gguf models/mnist/t10k-images.idx3-ubyte",
                                                 env::args().nth(0).expect("executable name should be defined")));
        }
    };


    // load the model
    let model = time!(mnist_model_load(model_file))?;

    // read a random digit from the test set
    let mut buf: [u8; 784] = [0; 784];
    let mut digit: [f32; 784] = [0f32; 784];
    {
        let mut fin = File::open(test_set_file)?;

        // seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
        fin.seek(SeekFrom::Start((16 + 784 * (rand::random::<usize>() % 10000)) as u64))?;
        fin.read_exact(&mut buf)?;
    }

    // render the digit in ASCII
    {
        for row in 0..28 {
            for col in 0..28 {
                eprint!("{} ", if buf[row * 28 + col] > 230 { '*' } else { '_' });
                digit[row * 28 + col] = buf[row * 28 + col] as f32;
            }

            eprintln!();
        }

        eprintln!();
    }

    let prediction = mnist_eval(&model, 1, &digit, Some("mnist.ggml"))?;
    println!("predicted digit is {}", prediction);

    Ok(())
}